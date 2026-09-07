"""Causal model-priced delta, band and two-sensitivity controls.

These transparent controls hedge the liability's partial spot derivative at
the currently observed variance. The stock-plus-option control additionally
matches the variance derivative before common holding caps are applied. They
do not optimize terminal ES. Bands are explicit stock-quantity half-widths and
must be chosen on development data before comparative evaluation.
"""

from __future__ import annotations

import math

import torch

from hedging_gym.finance import MarketBank, option_price


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


def spot_variance_greeks(spot, variance, time_index, config, *, hedge_index=0, chunk_size=1024):
    """Return (dS,dv), each shaped [..., liability/selected hedge], in float64.

    The signed liability and selected unit hedge option get their model-price
    spot and observed-variance derivatives.
    This is d/dv, not d/dsqrt(v). Retain the raw derivatives, including numerical
    roundoff near zero. Select the hedge contract before evaluating controls.
    """
    return _pair_greeks(spot, variance, time_index, config,
                        hedge_index=hedge_index, chunk_size=chunk_size)


def spot_gamma_greeks(spot, variance, time_index, config, *, hedge_index=0, chunk_size=1024):
    """Model-price (delta, spot gamma), holding current variance fixed."""
    return _pair_greeks(spot, variance, time_index, config,
                        hedge_index=hedge_index, chunk_size=chunk_size, gamma=True)


def _pair_greeks(spot, variance, time_index, config, *, hedge_index, chunk_size, gamma=False):
    if not 0 <= hedge_index < len(config.portfolio.hedges) or chunk_size < 1:
        raise ValueError("select an available hedge option and a positive chunk size")
    liability = config.portfolio.liability
    hedge = config.portfolio.hedges[hedge_index]
    s, v, t = torch.broadcast_tensors(
        spot.detach().to(torch.float64),
        torch.as_tensor(variance, dtype=torch.float64, device=spot.device).detach(),
        torch.as_tensor(time_index, dtype=torch.float64, device=spot.device),
    )
    shape = s.shape
    s, v, time = s.reshape(-1), v.reshape(-1), t.reshape(-1) * config.dt
    maturities = torch.stack((liability.maturity - time, hedge.maturity - time), dim=-1)
    if bool((maturities <= 0).any()):
        raise ValueError("hedge Greeks require decisions strictly before settlement")
    deltas, dvariances = [], []
    for offset in range(0, s.numel(), chunk_size):
        sl = slice(offset, offset + chunk_size)
        with torch.enable_grad():
            ss = s[sl, None].expand(-1, 2).clone().requires_grad_(True)
            vv = v[sl, None].expand(-1, 2).clone().requires_grad_(True)
            # Pricing each contract through the shared option dispatcher keeps
            # mixed call/put books and their gradients on the core equations.
            prices = torch.stack((
                config.portfolio.liability_quantity * option_price(
                    ss[:, 0], vv[:, 0], maturities[sl, 0], liability.strike,
                    config.market, kind=liability.kind),
                option_price(ss[:, 1], vv[:, 1], maturities[sl, 1], hedge.strike,
                             config.market, kind=hedge.kind),
            ), dim=-1)
            if gamma:
                ds, = torch.autograd.grad(prices.sum(), ss, create_graph=True)
                dv, = torch.autograd.grad(ds.sum(), ss)
            else:
                ds, dv = torch.autograd.grad(prices.sum(), (ss, vv))
        deltas.append(ds.detach())
        dvariances.append(dv.detach())
    return (torch.cat(deltas).reshape(shape + (2,)),
            torch.cat(dvariances).reshape(shape + (2,)))


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
    if not 0 <= hedge_index < len(bank.config.portfolio.hedges):
        raise ValueError("select an available hedge option")
    c = bank.config
    lower = c.execution.vector("holding_lower", c.n_assets)
    upper = c.execution.vector("holding_upper", c.n_assets)
    if greeks is None:
        times = torch.arange(c.n_steps, device=bank.spot.device)[None, :]
        greeks = spot_variance_greeks(bank.spot[:, :-1], bank.variance[:, :-1], times, c,
                                      hedge_index=hedge_index, chunk_size=chunk_size)
    ds, dv = greeks
    if ds.shape != bank.spot[:, :-1].shape + (2,) or dv.shape != ds.shape:
        raise ValueError("cached Greeks must have shape [paths,n_steps,2]")
    if not bool(torch.isfinite(ds).all() and torch.isfinite(dv).all() and (dv[..., 1] > 0).all()):
        raise ValueError("hedge variance sensitivity needs pricing-domain requalification")
    instrument = hedge_index + 1
    hedge = (dv[..., 0] / dv[..., 1]).clamp(lower[instrument], upper[instrument])
    stock = (ds[..., 0] - hedge * ds[..., 1]).clamp(lower[0], upper[0])
    positions = stock.new_zeros((*stock.shape, c.n_assets))
    positions[..., 0] = stock
    positions[..., instrument] = hedge
    return positions.to(bank.spot.dtype)


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
    return delta_variance_hedge_positions(bank, hedge_index, greeks, chunk_size=chunk_size)
