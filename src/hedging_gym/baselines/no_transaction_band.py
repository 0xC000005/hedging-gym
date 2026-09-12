"""Learn when not to trade using no-transaction bands.

Paper: Imaki et al., https://arxiv.org/abs/2103.01775v1.
Reference: https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/README.md.
Implementation: local PyTorch transfer with a learned center and per-instrument
bounded widths. See docs/baseline-methods.md for the financial mapping.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from hedging_gym.baselines._shared.controllers import (
    policy_controller as make_controller,
)
from hedging_gym.baselines._shared.pathwise import train_pathwise
from hedging_gym.baselines._shared.policy import (
    BUY,
    HOLD,
    SELL,
    PolicyAction,
    _bounds,
    _ConfiguredPolicy,
    _network,
)
from hedging_gym.environment.config import HedgingConfig

__all__ = ["NoTransactionBandPolicy", "train", "make_controller"]


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


def train(train_bank, **options):
    """Train no-transaction bands; options are documented in train_pathwise."""
    return train_pathwise(NoTransactionBandPolicy, train_bank, method="ntb",
                          method_label="Learned no-transaction bands", **options)
