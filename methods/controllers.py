"""Adapt existing baselines to the core's causal controller interface.

Controllers receive the current observation and ledger, never the market bank.
Targets reach the common environment unchanged; infeasible trades raise.
"""

from contextlib import contextmanager
import math

import torch

from .classical import spot_delta, spot_gamma_greeks, spot_variance_greeks


@contextmanager
def _evaluation_mode(module):
    training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training)


def policy_controller(policy):
    """Deterministic DH/NTB target adapter, preserving the module's prior mode."""
    def control(observed, ledger, time_index, config):
        with _evaluation_mode(policy):
            return policy(observed, ledger.positions, config.holding_lower,
                          config.holding_upper, deterministic=True).target_holdings
    control.action_selection = "deterministic"
    return control


def classical_controller(method="delta_gamma", *, band=0., hedge_index=0, chunk_size=1024):
    """Current-state source Greeks with the existing bounded hedge algebra.

    Delta-only supports the source stock-quantity band; spot gamma and variance
    sensitivity are distinct options. Future marks never enter Greek inputs.
    """
    if method not in {"delta", "delta_gamma", "delta_variance"}:
        raise ValueError("unknown classical hedge")
    if not math.isfinite(band) or band < 0 or (method != "delta" and band):
        raise ValueError("only delta hedging accepts a nonnegative stock band")
    def control(observed, ledger, time_index, config):
        spot, variance = observed[:, 1] * config.spot0, observed[:, 2] * max(config.v0, 1e-4)
        if method == "delta":
            delta = spot_delta(spot, variance, time_index, config, chunk_size=chunk_size)
            lower = (delta-band).clamp(config.holding_lower[0], config.holding_upper[0])
            upper = (delta+band).clamp(config.holding_lower[0], config.holding_upper[0])
            targets = torch.zeros_like(ledger.positions)
            targets[:, 0] = torch.minimum(torch.maximum(ledger.positions[:, 0], lower), upper)
            return targets
        greek = spot_gamma_greeks if method == "delta_gamma" else spot_variance_greeks
        ds, exposure = greek(spot, variance, time_index, config, hedge_index=hedge_index, chunk_size=chunk_size)
        if not bool(torch.isfinite(ds).all() and torch.isfinite(exposure).all() and (exposure[:, 1] > 0).all()):
            raise ValueError("hedge sensitivity needs pricing-domain requalification")
        instrument = hedge_index + 1
        call = (exposure[:, 0]/exposure[:, 1]).clamp(config.holding_lower[instrument], config.holding_upper[instrument])
        stock = (ds[:, 0]-call*ds[:, 1]).clamp(config.holding_lower[0], config.holding_upper[0])
        targets = torch.zeros_like(ledger.positions)
        targets[:, 0], targets[:, instrument] = stock, call
        return targets
    control.action_selection = "deterministic"
    return control
