"""Bounded Deep Hedging and learned no-transaction-band policies.

See methods/README.md for method attribution. These continuous policies do not
parameterize lot sizes or minimum orders, or estimate fixed-ticket gate gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from hedging_gym.config import HedgingConfig
from hedging_gym.finance import instrument_names, observation_fields


HOLD, BUY, SELL = 0, 1, 2


@dataclass(frozen=True)
class PolicyAction:
    target_holdings: Tensor  # [batch, assets]
    modes: Tensor  # [batch, assets], HOLD=0, BUY=1, SELL=2
    log_prob: Tensor  # [batch], joint log probability across assets
    probabilities: Tensor | None = None  # [batch, assets, 3]
    entropy: Tensor | None = None  # [batch], summed across assets


def _network(feature_dim: int, output_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    if feature_dim < 1 or output_dim < 1 or any(width < 1 for width in hidden):
        raise ValueError("network dimensions must be positive")
    layers: list[nn.Module] = []
    previous = feature_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def _bounds(
    features: Tensor, holdings: Tensor, lower, upper, feature_dim: int, n_assets: int,
) -> tuple[Tensor, Tensor]:
    if features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError(f"features must have shape [batch, {feature_dim}]")
    if holdings.shape != (features.shape[0], n_assets):
        raise ValueError(f"holdings must have shape [batch, {n_assets}]")
    lo = torch.as_tensor(-torch.inf if lower is None else lower, dtype=holdings.dtype, device=holdings.device)
    hi = torch.as_tensor(torch.inf if upper is None else upper, dtype=holdings.dtype, device=holdings.device)
    # Finance validates the bounds and input holdings once. This hot path
    # checks shape without introducing a per-date accelerator synchronization.
    return torch.broadcast_to(lo, holdings.shape), torch.broadcast_to(hi, holdings.shape)


class _ConfiguredPolicy(nn.Module):
    """Policy geometry follows the environment's ordered financial schema."""

    def __init__(self, config: HedgingConfig):
        super().__init__()
        if not isinstance(config, HedgingConfig):
            raise TypeError("construct a policy with HedgingConfig or use from_env(env)")
        self.observation_fields = tuple(observation_fields(config))
        self.instrument_names = tuple(instrument_names(config))
        self.feature_dim = len(self.observation_fields)
        self.n_assets = config.n_assets

    @classmethod
    def from_env(cls, env, **kwargs):
        """Build from the same config used by a scalar or batched environment."""
        return cls(env.config, **kwargs)

    def check_config(self, config):
        """Allow observed parameter changes, rejecting different field meanings."""
        if (tuple(observation_fields(config)) != self.observation_fields
                or tuple(instrument_names(config)) != self.instrument_names):
            raise ValueError("policy observation/instrument schema differs from the environment config")

    def get_extra_state(self):
        """Keep field meanings with saved weights, including same-width books."""
        return dict(observation_fields=self.observation_fields,
                    instrument_names=self.instrument_names)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("checkpoint observation/instrument schema differs from this policy")


class DirectDHPolicy(_ConfiguredPolicy):
    """Direct Deep Hedging with smooth bounded absolute target holdings.

    Pass the shared causal observation features, including current
    cash, holdings and declared market/cost parameters. The returned mode is a
    description of the realized trade, not a trainable discrete distribution.
    A fixed-ticket ledger therefore gives this policy the usual conditional
    pathwise gradient; it provides no estimator for learning a hard ticket gate.
    """

    def __init__(
        self, config: HedgingConfig, hidden: Sequence[int] = (64, 64),
    ):
        super().__init__(config)
        self.continuous = _network(self.feature_dim, self.n_assets, hidden)

    def forward(
        self, features: Tensor, holdings: Tensor, lower, upper, *,
        deterministic: bool = True, generator: torch.Generator | None = None,
    ) -> PolicyAction:
        del deterministic, generator  # Both policies share the rollout interface.
        lo, hi = _bounds(features, holdings, lower, upper, self.feature_dim, self.n_assets)
        raw = self.continuous(features)
        finite = lo.isfinite() & hi.isfinite()
        safe_lo, safe_hi = torch.where(finite, lo, 0.), torch.where(finite, hi, 0.)
        bounded = safe_lo + (safe_hi - safe_lo) * raw.sigmoid()
        target = torch.where(finite, bounded, raw).clamp(min=lo, max=hi)
        modes = torch.where(
            target > holdings, BUY, torch.where(target < holdings, SELL, HOLD),
        )
        return PolicyAction(target, modes, features.new_zeros(features.shape[0]))


class NoTransactionBandPolicy(_ConfiguredPolicy):
    """Learn bounded bands and clamp previous holdings into them, giving hold.

    This adapts the established PFHedge/Imaki no-transaction-band construction.
    A learned center replaces the single-stock Black-Scholes anchor, and
    separate bounded widths handle all assets. It uses the same causal state
    as the other policies. A fixed ticket still creates a discontinuity at
    the band boundary: ordinary pathwise training omits that boundary term.
    No straight-through gradient is used or claimed to repair it.
    """

    def __init__(
        self, config: HedgingConfig, hidden: Sequence[int] = (64, 64),
        *, initial_width_logit: float = -3.0,
    ):
        super().__init__(config)
        if config.execution.holding_lower is None or config.execution.holding_upper is None:
            raise ValueError("this band parameterization requires finite holding bounds")
        self.continuous = _network(self.feature_dim, 3 * self.n_assets, hidden)
        # Narrow initial bands avoid an initial always-hold policy with zero
        # pathwise learning signal. Tune the width on development data.
        with torch.no_grad():
            self.continuous[-1].bias.reshape(self.n_assets, 3)[:, 1:] = initial_width_logit

    def bands(
        self, features: Tensor, holdings: Tensor, lower, upper,
    ) -> tuple[Tensor, Tensor]:
        lo, hi = _bounds(features, holdings, lower, upper, self.feature_dim, self.n_assets)
        fractions = self.continuous(features).reshape(-1, self.n_assets, 3).sigmoid()
        center = lo + (hi - lo) * fractions[..., 0]
        band_lower = center - (center - lo) * fractions[..., 1]
        band_upper = center + (hi - center) * fractions[..., 2]
        return band_lower, band_upper

    def forward(
        self, features: Tensor, holdings: Tensor, lower, upper, *,
        deterministic: bool = True, generator: torch.Generator | None = None,
    ) -> PolicyAction:
        del deterministic, generator
        band_lower, band_upper = self.bands(features, holdings, lower, upper)
        target = torch.clamp(holdings, min=band_lower, max=band_upper)
        modes = torch.where(
            target > holdings, BUY, torch.where(target < holdings, SELL, HOLD),
        )
        return PolicyAction(target, modes, features.new_zeros(features.shape[0]))
