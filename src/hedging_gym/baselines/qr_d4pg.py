"""QR-D4PG adaptation of Cao et al.'s distributional hedger.

Paper: Gamma and Vega Hedging Using Deep Distributional Reinforcement Learning.
https://doi.org/10.3389/frai.2023.1129370
Author source: agent/learning.py and agent/distributional.py at the pinned URL
below. The local PyTorch critic retains quantile targets and deterministic
actor gradients, but uses uniform replay, Polyak targets and the common global
terminal-ES objective instead of native conditional VaR/CVaR. It is not the
authors' TensorFlow option-arrival reproduction. The shared learner retains
its upstream Apache-2.0 notice. See docs/baseline-methods.md.
"""
import torch
from torch import nn

from hedging_gym.baselines._shared.controllers import (
    policy_controller as make_controller,
)
from hedging_gym.baselines._shared.d4pg import train_d4pg
from hedging_gym.baselines._shared.policy import _network

SOURCE = "https://github.com/rotmanfinhub/gamma-vega-rl-hedging/tree/77dc48326da000d983b1fb750edb2177e38c75fd"

__all__ = ["QuantileCritic", "train", "make_controller", "SOURCE"]


class QuantileCritic(nn.Module):
    """Conditional loss quantiles, measured in standardized loss units."""

    def __init__(self, feature_dim, n_assets, hidden=(64, 64), *, quantiles=128):
        super().__init__()
        self.quantiles = _network(feature_dim+n_assets, quantiles, hidden)
        self.tail = None
        self.tail_threshold = None
        self.register_buffer("probabilities", (torch.arange(quantiles)+.5)/quantiles)
        edges = torch.arange(quantiles+1)/quantiles
        self.register_buffer("body_weights", (edges[1:].clamp_max(1.)
                                             - edges[:-1].clamp_max(1.)))

    def forward(self, observed, actions):
        inputs = torch.cat((observed, actions), -1)
        values = self.quantiles(inputs)
        return values, None, None

    def distribution(self, observed, actions):
        """Equal-probability quadrature for the target loss distribution."""
        values, _, _ = self(observed, actions)
        return values.sort(-1).values

    def expected_ru(self, observed, actions, threshold, alpha):
        """Expected global terminal-risk loss under the learned quantiles."""
        values, _, _ = self(observed, actions)
        values = values.sort(-1).values
        threshold = torch.as_tensor(threshold, device=values.device, dtype=values.dtype)
        threshold_values = threshold[:, None] if threshold.ndim else threshold
        positive_part = ((values-threshold_values).clamp_min(0)*self.body_weights).sum(-1)
        return threshold + positive_part/(1-alpha)


def train(train_bank, **options):
    """Train QR-D4PG using the common collector and distributional update loop."""
    # The historical artifact key hull_rl refers to this 2023 QR-D4PG port,
    # not the separate two-moment Hull DDPG baseline.
    return train_d4pg("hull_rl", train_bank, critic_class=QuantileCritic,
                      source=SOURCE, **options)
