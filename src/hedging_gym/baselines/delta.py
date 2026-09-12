"""Current-state delta hedging, with an optional stock-quantity band.

Implementation: local sensitivity matching, not a learned trading strategy.
Pricing and benchmark references: docs/paper-benchmarks.md and docs/validation.md.
"""
from __future__ import annotations

import math

import torch

from hedging_gym.environment.finance import (
    MarketBank,
    decode_market_observation,
    option_price,
)
from hedging_gym.interfaces import Controller


def spot_delta(spot, variance, time_index, config, *, chunk_size=1024):
    """Partial spot derivative of the signed option liability, at fixed variance.

    Each row uses only that row's current spot, observed variance and time.
    Flattening independent rows batches identical computations; it does not
    expose later prices to an earlier decision. Chunked float64 autograd keeps
    pricing accuracy and bounds memory. Outputs use the input spot dtype;
    roundoff is projected into the option's r=q=0 unit-delta bounds before
    applying the signed liability quantity.
    """
    if chunk_size < 1:
        raise ValueError("Greek pricing chunks must be nonempty")
    reference = spot.detach().to(torch.float64)
    liability = config.portfolio.liability
    time = torch.as_tensor(time_index, dtype=torch.float64, device=spot.device)
    spot_values, variance_values, maturities = torch.broadcast_tensors(
        reference, torch.as_tensor(variance, dtype=torch.float64, device=spot.device).detach(),
        liability.maturity - time * config.dt,
    )
    if bool((maturities <= 0).any()):
        raise ValueError("hedge deltas require a decision strictly before liability settlement")
    shape = spot_values.shape
    spot_values, variance_values, maturities = (x.reshape(-1) for x in
                                              (spot_values, variance_values, maturities))
    result = []
    for offset in range(0, spot_values.numel(), chunk_size):
        sl = slice(offset, offset + chunk_size)
        with torch.enable_grad():
            current_spot = spot_values[sl].clone().requires_grad_(True)
            price = option_price(current_spot, variance_values[sl], maturities[sl],
                                 liability.strike, config.market, kind=liability.kind)
            delta, = torch.autograd.grad(price.sum(), current_spot)
        result.append(delta.detach())
    lower, upper = (-1.0, 0.0) if liability.kind == "put" else (0.0, 1.0)
    delta = torch.cat(result).reshape(shape).clamp(lower, upper)
    return (config.portfolio.liability_quantity * delta).to(spot.dtype)


def delta_hedge_positions(bank: MarketBank, *, band: float = 0.0, deltas=None, chunk_size=1024):
    """Stock-only holdings [paths,dates,n_assets], under the common limits.

    Zero band is projected delta hedging. With positive band, previous stock
    holdings are clamped into [delta-band,delta+band] intersected with the
    allowed interval: hold inside, trade to the nearest edge outside. All
    prices, fees and final liquidation come from ``ledger_from_positions``.

    Supplying the same cached current-state deltas makes development band
    comparisons cheap; it does not change their financial information.
    """
    if band < 0 or not math.isfinite(band):
        raise ValueError("band must be a finite nonnegative stock quantity")
    c = bank.config
    holding_lower = c.execution.vector("holding_lower", c.n_assets)
    holding_upper = c.execution.vector("holding_upper", c.n_assets)
    if deltas is None:
        times = torch.arange(c.n_steps, device=bank.spot.device)[None, :]
        deltas = spot_delta(bank.spot[:, :-1], bank.variance[:, :-1], times, c,
                            chunk_size=chunk_size)
    if deltas.shape != bank.spot[:, :-1].shape:
        raise ValueError("cached deltas must have shape [paths,n_steps]")
    lower = (deltas - band).clamp(holding_lower[0], holding_upper[0])
    upper = (deltas + band).clamp(holding_lower[0], holding_upper[0])
    previous = torch.zeros_like(bank.spot[:, 0])
    positions = []
    for t in range(c.n_steps):
        previous = torch.minimum(torch.maximum(previous, lower[:, t]), upper[:, t])
        zeros = previous.new_zeros((*previous.shape, c.n_assets - 1))
        positions.append(torch.cat((previous[..., None], zeros), dim=-1))
    return torch.stack(positions, dim=1)


def make_controller(*, band=0., chunk_size=1024) -> Controller:
    """Hold inside the configured delta band; otherwise trade to its edge."""
    if not math.isfinite(band) or band < 0:
        raise ValueError("delta band must be finite and nonnegative")

    def control(observed, ledger, time_index, config):
        spot, variance = decode_market_observation(observed, config)
        lower = config.execution.vector("holding_lower", config.n_assets)
        upper = config.execution.vector("holding_upper", config.n_assets)
        delta = spot_delta(spot, variance, time_index, config, chunk_size=chunk_size)
        band_lower = (delta-band).clamp(lower[0], upper[0])
        band_upper = (delta+band).clamp(lower[0], upper[0])
        targets = torch.zeros_like(ledger.positions)
        targets[:, 0] = torch.minimum(torch.maximum(ledger.positions[:, 0], band_lower), band_upper)
        return targets

    control.action_selection = "deterministic"
    return control
