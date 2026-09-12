"""Delta/variance sensitivity matching with stock and a selected hedge.

Implementation: local model-priced sensitivity control, not terminal-risk
optimization. See docs/paper-benchmarks.md and docs/validation.md for references.
"""
import torch

from hedging_gym.baselines._shared.greeks import (
    _pair_greeks,
    _two_sensitivity_positions,
    two_sensitivity_controller,
)
from hedging_gym.environment.finance import MarketBank
from hedging_gym.interfaces import Controller


def spot_variance_greeks(spot, variance, time_index, config, *, hedge_index=0, chunk_size=1024,
                         integrated_variance=None):
    """Return (dS,dv), each shaped [..., liability/selected hedge], in float64.

    The signed liability and selected unit hedge option get their model-price
    spot and observed-variance derivatives.
    This is d/dv, not d/dsqrt(v). Retain the raw derivatives, including numerical
    roundoff near zero. Select the hedge contract before evaluating controls.
    """
    return _pair_greeks(spot, variance, time_index, config,
                        hedge_index=hedge_index, chunk_size=chunk_size, integrated_variance=integrated_variance)


def delta_variance_hedge_positions(bank: MarketBank, hedge_index=0, greeks=None, *, chunk_size=1024):
    """Stock plus one preselected option, matching dS and dv under shared caps.

    First q_hedge=F_v/C_v is projected to its allowed holdings; then
    q_stock=F_S-q_hedge*C_S is recomputed and projected to the stock bounds.
    Other hedge options are unused. This is sensitivity matching under caps and daily
    trading, not an ES-optimal policy or a replication claim for jump markets.
    In GBM, dv is sensitivity to the constant variance parameter.

    No denominator floor or favorable near-zero trade threshold is applied.
    Nonpositive/nonfinite hedge variance sensitivities require requalification
    rather than a silent replacement. Cached Greeks must match this bank and
    the preselected hedge. Charge their full computation in cost comparisons.
    """
    c = bank.config
    if greeks is None:
        times = torch.arange(c.n_steps, device=bank.spot.device)[None, :]
        greeks = spot_variance_greeks(bank.spot[:, :-1], bank.variance[:, :-1], times, c,
                                      hedge_index=hedge_index, chunk_size=chunk_size,
                                      integrated_variance=None if bank.integrated_variance is None
                                      else bank.integrated_variance[:, :-1])
    return _two_sensitivity_positions(bank, hedge_index, greeks)


def make_controller(*, hedge_index=0, chunk_size=1024) -> Controller:
    """Construct a causal delta/variance controller for the selected hedge."""
    return two_sensitivity_controller(spot_variance_greeks,
                                      hedge_index=hedge_index, chunk_size=chunk_size)
