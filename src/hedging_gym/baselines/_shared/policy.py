"""Shared action records and configured neural-policy building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from hedging_gym.environment.config import HedgingConfig
from hedging_gym.environment.finance import instrument_names, observation_fields

HOLD, BUY, SELL = 0, 1, 2


@dataclass(frozen=True)
class PolicyAction:
    target_holdings: Tensor  # [batch, assets]
    modes: Tensor  # [batch, assets], HOLD=0, BUY=1, SELL=2
    log_prob: Tensor  # [batch], joint log probability across assets
    probabilities: Tensor | None = None  # [batch, joint_modes], when supplied
    entropy: Tensor | None = None  # [batch], entropy of the joint action


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
