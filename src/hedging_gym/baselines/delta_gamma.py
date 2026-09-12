"""Delta/gamma sensitivity matching with stock and a selected hedge.

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


def spot_gamma_greeks(spot, variance, time_index, config, *, hedge_index=0, chunk_size=1024):
    """Model-price (delta, spot gamma), holding current variance fixed."""
    return _pair_greeks(spot, variance, time_index, config,
                        hedge_index=hedge_index, chunk_size=chunk_size, gamma=True)


def delta_gamma_hedge_positions(bank: MarketBank, hedge_index=0, greeks=None, *, chunk_size=1024):
    """Match liability delta/gamma with stock/option, under the shared bounds.

    This is distinct from matching the variance exposure. Both controls
    use the same two-sensitivity hedge algebra; neither is ES optimized.
    """
    if greeks is None:
        times = torch.arange(bank.config.n_steps, device=bank.spot.device)[None, :]
        greeks = spot_gamma_greeks(bank.spot[:, :-1], bank.variance[:, :-1],
                                   times, bank.config, hedge_index=hedge_index,
                                   chunk_size=chunk_size)
    return _two_sensitivity_positions(bank, hedge_index, greeks)


def make_controller(*, hedge_index=0, chunk_size=1024) -> Controller:
    """Construct a causal delta/gamma controller for the selected hedge."""
    return two_sensitivity_controller(spot_gamma_greeks,
                                      hedge_index=hedge_index, chunk_size=chunk_size)
