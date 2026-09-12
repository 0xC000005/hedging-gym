"""Deep Hedging: direct, bounded target holdings learned through accounting.

Paper: Bühler et al., https://arxiv.org/abs/1802.03042v1.
Implementation: local PyTorch implementation of pathwise policy learning, not
imported author code. Configuration and source differences: docs/baseline-methods.md.
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

__all__ = ["DirectDHPolicy", "train", "make_controller"]


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


def train(train_bank, **options):
    """Train direct Deep Hedging; options are documented in train_pathwise."""
    return train_pathwise(DirectDHPolicy, train_bank, method="dh",
                          method_label="Deep Hedging", **options)
