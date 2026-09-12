"""EX-DRL-inspired EX-D4PG with a fitted generalized Pareto loss tail.

Paper: Malekzadeh et al., EX-DRL.
https://arxiv.org/abs/2408.12446
Author source: agent/learning.py and agent/distributional.py at SOURCE below.
The tail participates in both targets and actor improvement; it is not market
data. This PyTorch port uses the common global terminal-ES objective, uniform
replay and Polyak targets. Inverse-CDF quadrature and analytic tail expectations
replace sampled tail integration. See docs/baseline-methods.md for the native
author learner and the separate common-environment port qualification.
"""
import math

import torch
from torch.nn import functional as F

from hedging_gym.baselines._shared.controllers import (
    policy_controller as make_controller,
)
from hedging_gym.baselines._shared.d4pg import train_d4pg
from hedging_gym.baselines._shared.policy import _network

from .qr_d4pg import QuantileCritic

SOURCE = "https://github.com/pmalekzadeh/EX-DRL/tree/f1abe99df7fa9efaa65af6b9dd416c3425c64098"

__all__ = ["ParetoCritic", "train", "make_controller", "gpd_expected_excess", "gpd_nll", "SOURCE"]


def gpd_expected_excess(cutoff, scale, shape):
    """E[(Y-cutoff)+] for Y~GPD(scale, shape), 0<shape<1.

    Integrating the survival function avoids the author's rejection-sampling
    loop for the tail expectation. This is the same Pareto distribution, not a
    Gaussian or clipped-quantile substitute.
    """
    positive = cutoff.clamp_min(0)
    survival = torch.exp(-torch.log1p(shape * positive / scale) / shape)
    return survival * (scale + shape * positive) / (1-shape) + (-cutoff).clamp_min(0)


def gpd_nll(excess, scale, shape):
    """Negative log-likelihood of nonnegative Pareto excess observations."""
    return torch.log(scale) + (1 + 1/shape) * torch.log1p(shape * excess / scale)


class ParetoCritic(QuantileCritic):
    """Quantile body plus a learned finite-mean generalized Pareto tail."""

    def __init__(self, feature_dim, n_assets, hidden=(64, 64), *, quantiles=128,
                 tail_threshold=.96):
        super().__init__(feature_dim, n_assets, hidden, quantiles=quantiles)
        self.tail = _network(feature_dim+n_assets, 2, hidden)
        self.tail_threshold = tail_threshold
        # Integrate the body only below the splice; the GPD supplies its mass.
        edges = torch.arange(quantiles+1)/quantiles
        self.body_weights = (edges[1:].clamp_max(tail_threshold)
                             - edges[:-1].clamp_max(tail_threshold))

    def forward(self, observed, actions):
        inputs = torch.cat((observed, actions), -1)
        values = self.quantiles(inputs)
        raw = self.tail(inputs)
        epsilon = torch.finfo(raw.dtype).eps
        scale = F.softplus(raw[:, 0]) + epsilon
        # Author heavy_tail=True also restricts shape to (0,1), finite-mean GPD.
        shape = raw[:, 1].sigmoid().clamp(epsilon, 1-epsilon)
        return values, scale, shape

    def threshold_value(self, values):
        values = values.sort(-1).values
        index = self.tail_threshold * values.shape[-1] - .5
        left = min(max(math.floor(index), 0), values.shape[-1]-1)
        right = min(left+1, values.shape[-1]-1)
        return values[:, left] + (index-left) * (values[:, right]-values[:, left])

    def distribution(self, observed, actions):
        """Equal-probability quadrature for the target loss distribution."""
        values, scale, shape = self(observed, actions)
        values = values.sort(-1).values
        location = self.threshold_value(values)
        tail_probabilities = ((self.probabilities-self.tail_threshold)
                              / (1-self.tail_threshold)).clamp_min(0)
        excess = (scale[:, None] / shape[:, None]
                  * torch.expm1(-shape[:, None] * torch.log1p(-tail_probabilities)))
        return torch.where(self.probabilities[None] > self.tail_threshold,
                           location[:, None]+excess, values)

    def expected_ru(self, observed, actions, threshold, alpha):
        values, scale, shape = self(observed, actions)
        values = values.sort(-1).values
        threshold = torch.as_tensor(threshold, device=values.device, dtype=values.dtype)
        threshold_values = threshold[:, None] if threshold.ndim else threshold
        positive_part = ((values-threshold_values).clamp_min(0)*self.body_weights).sum(-1)
        location = self.threshold_value(values)
        positive_part = positive_part + (1-self.tail_threshold) * gpd_expected_excess(
            threshold-location, scale, shape)
        return threshold + positive_part/(1-alpha)

    def tail_loss(self, observed, actions):
        """Fit excess upper quantiles by MLE, as in the EX-DRL author learner."""
        values, scale, shape = self(observed, actions)
        values = values.detach().sort(-1).values
        excess = (values[:, self.probabilities > self.tail_threshold]
                  - self.threshold_value(values)[:, None]).clamp_min(0)
        return gpd_nll(excess, scale[:, None], shape[:, None]).mean()


def train(train_bank, **options):
    """Train the quantile/GPD hedger using the common D4PG update machinery."""
    return train_d4pg("exdrl", train_bank, critic_class=ParetoCritic,
                      source=SOURCE, **options)
