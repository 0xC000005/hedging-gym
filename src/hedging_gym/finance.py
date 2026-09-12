"""Batched market prices and paths with one differentiable portfolio cash ledger.

Heston uses quadratic-exponential simulation, with a martingale correction by
default (QE-M) or legacy plain QE. Both approximate continuous-time Heston.
GBM is exact at trading dates; Bates adds compensated independent jumps.
Independent references live in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch

from .config import (
    HedgingConfig, HestonConfig, TimeGrid, EXECUTION_FIELDS,
)

@lru_cache(maxsize=None)
def _quadrature(order: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre quadrature on one Fourier integration panel."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    upper = 150.0
    return (nodes + 1.0) * upper / 2.0, weights * upper / 2.0


@dataclass
class MarketBank:
    spot: torch.Tensor                 # [paths, n_steps + 1]
    variance: torch.Tensor
    marks: torch.Tensor                # [paths, n_steps + 1, n_assets]
    liability: torch.Tensor            # signed value of configured liability
    config: HedgingConfig
    integrated_variance: torch.Tensor | None = None


BANK_FIELDS = ("spot", "variance", "marks", "liability")


def bank_to(bank: MarketBank, device) -> MarketBank:
    return MarketBank(*(getattr(bank, key).to(device) for key in BANK_FIELDS), bank.config,
                      None if bank.integrated_variance is None else bank.integrated_variance.to(device))


def bank_subset(bank: MarketBank, indices) -> MarketBank:
    return MarketBank(*(getattr(bank, key)[indices] for key in BANK_FIELDS), bank.config,
                      None if bank.integrated_variance is None else bank.integrated_variance[indices])


@dataclass
class LedgerState:
    cash: torch.Tensor
    positions: torch.Tensor
    total_cost: torch.Tensor
    turnover: torch.Tensor              # per-instrument quantities, including liquidation
    tickets: torch.Tensor               # count, even in cases where ticket price is zero


def _tensor(value, reference):
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _characteristic(u, log_spot, variance, maturity, c):
    b = c.kappa - c.rho * c.sigma * 1j * u
    d = torch.sqrt(b * b + c.sigma**2 * (u * u + 1j * u))
    d = torch.where(d.real < 0, -d, d)
    g = (b - d) / (b + d)
    exp_dt = torch.exp(-d * maturity)
    constant = 1j * u * log_spot + c.kappa * c.theta / c.sigma**2 * (
        (b - d) * maturity - 2 * torch.log((1 - g * exp_dt) / (1 - g))
    )
    coefficient = (b - d) / c.sigma**2 * (1 - exp_dt) / (1 - g * exp_dt)
    result = torch.exp(constant + coefficient * variance)
    if c.model == "bates":
        from .bates import jump_characteristic
        result = result * jump_characteristic(u, maturity, c)
    return result


def heston_call_price(spot, variance, maturity, strike, config: HestonConfig):
    """Heston/Bates Fourier price; all quadrature runs in float64.

    Use 96-node [0,150] panels, extending the cutoff for low integrated
    variance and slow characteristic-function decay at high vol-of-vol.
    The exponential tail scale follows QuantLib's AnalyticHestonEngine;
    see docs/validation.md. This is a numerical rule checked against QuantLib,
    not a universal error bound.
    This function preserves gradients with respect to tensor inputs. Price-bank
    construction chunks calls to bound peak memory.
    """
    if not isinstance(spot, torch.Tensor):
        spot = torch.as_tensor(spot, dtype=torch.float64)
    result_dtype = spot.dtype
    reference = spot.to(torch.float64)
    s, v, tau, k = torch.broadcast_tensors(
        reference, _tensor(variance, reference), _tensor(maturity, reference), _tensor(strike, reference),
    )
    if bool(((s <= 0) | (v < 0) | (tau < 0) | (k <= 0)).any()):
        raise ValueError("Heston requires positive spot/strike and nonnegative variance/time")
    shape = s.shape
    s, v, tau, k = (x.reshape(-1) for x in (s, v, tau, k))
    intrinsic = (s - k).clamp_min(0)
    active = torch.nonzero(tau > 0, as_tuple=True)[0]
    if active.numel() == 0:
        return intrinsic.reshape(shape).to(result_dtype)
    expected_variance = config.theta * tau + (v - config.theta) * (
        -torch.expm1(-config.kappa * tau)
    ) / config.kappa
    cutoff = 12.0 / expected_variance.clamp_min(1e-20).sqrt()
    if abs(config.rho) < 1:
        # |phi(u)| decays asymptotically as exp(-decay*u). Expected variance
        # alone misses this slower tail in the original Deep Hedging setting.
        decay = math.sqrt(1 - config.rho**2) * (
            v + config.kappa * config.theta * tau
        ) / config.sigma
        cutoff = torch.maximum(cutoff, 32.0 / decay.clamp_min(1e-20))
    panels = torch.ceil(cutoff / 150.0)
    # Bucket nearby cutoffs together, without ever reducing the requested cutoff.
    buckets = torch.ceil(torch.log2(panels.clamp_min(1))).to(torch.int64)
    if bool((buckets[active] > 13).any()):
        raise ValueError("state needs more than 8192 Fourier panels; requalify its pricing domain")
    nodes_np, weights_np = _quadrature(96)
    nodes = _tensor(nodes_np, s)[:, None]
    weights = _tensor(weights_np, s)[:, None]
    result = intrinsic.clone()
    for bucket in torch.unique(buckets[active]).tolist():
        indices = active[buckets[active] == bucket]
        ss, vv, tt, kk = (x[indices][None, :] for x in (s, v, tau, k))
        p1, p2 = torch.zeros_like(ss), torch.zeros_like(ss)
        for panel in range(2**bucket):
            u = nodes + panel * 150.0
            oscillation = torch.exp(-1j * u * torch.log(kk))
            phi = _characteristic(u, torch.log(ss), vv, tt, config)
            shifted = _characteristic(u - 1j, torch.log(ss), vv, tt, config)
            p1 = p1 + (weights * (oscillation * shifted / (1j * u * ss)).real).sum(0)
            p2 = p2 + (weights * (oscillation * phi / (1j * u)).real).sum(0)
        value = (ss * (0.5 + p1 / math.pi) - kk * (0.5 + p2 / math.pi)).squeeze(0)
        result = result.index_copy(0, indices, torch.maximum(value, intrinsic[indices]))
    return result.reshape(shape).to(result_dtype)


def quantlib_option_price(spot, variance, maturity, strike, config, *, days_per_year=252, kind="call"):
    """Independent call/put reference on the selected 252/365/360 year clock.

    Contract times must map exactly to dates under that clock; no silent
    maturity rounding is used. Flat zero-rate curves match the shared ledger.
    """
    import QuantLib as ql

    if kind not in ("call", "put"):
        raise ValueError("choose call or put")
    clock = TimeGrid(days_per_year=days_per_year)
    if maturity == 0:
        return max((1 if kind == "call" else -1) * (float(spot) - float(strike)), 0.0)
    days = round(float(maturity) * clock.days_per_year)
    if days < 1 or not math.isclose(days * clock.dt, float(maturity), rel_tol=0, abs_tol=1e-12):
        raise ValueError("QuantLib reference requires maturities on the selected year clock")
    evaluation = ql.Date(2, ql.January, 2024)
    previous_evaluation = ql.Settings.instance().evaluationDate
    try:
        ql.Settings.instance().evaluationDate = evaluation
        day_count = {252: lambda: ql.Business252(ql.NullCalendar()),
                     365: ql.Actual365Fixed, 360: ql.Actual360}[days_per_year]()
        curve = ql.YieldTermStructureHandle(ql.FlatForward(evaluation, 0.0, day_count))
        quote = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
        if config.model == "gbm":
            vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
                evaluation, ql.NullCalendar(), math.sqrt(float(variance)), day_count))
            process = ql.BlackScholesMertonProcess(quote, curve, curve, vol)
            engine = ql.AnalyticEuropeanEngine(process)
        elif config.model == "bates":
            process = ql.BatesProcess(curve, curve, quote, float(variance), config.kappa,
                config.theta, config.sigma, config.rho, config.jump_intensity,
                config.jump_mean, config.jump_std)
            engine = ql.BatesEngine(ql.BatesModel(process), 192)
        else:
            process = ql.HestonProcess(curve, curve, quote, float(variance), config.kappa,
                                       config.theta, config.sigma, config.rho)
            engine = ql.AnalyticHestonEngine(ql.HestonModel(process), 1e-10, 10000)
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, float(strike)),
            ql.EuropeanExercise(evaluation + days),
        )
        option.setPricingEngine(engine)
        return float(option.NPV())
    finally:
        ql.Settings.instance().evaluationDate = previous_evaluation



def call_price(spot, variance, maturity, strike, config):
    """Black--Scholes for GBM; Heston/Bates characteristic-function pricing."""
    if config.model == "gbm":
        from .gbm import gbm_call_price
        return gbm_call_price(spot, variance, maturity, strike, config)
    return heston_call_price(spot, variance, maturity, strike, config)


def transition(spot, variance, shocks=None, config=None, *, dt, generator=None):
    """Shared conditional market interface; trading/accounting are separate."""
    config = config or HestonConfig()
    if config.model == "gbm":
        from .gbm import gbm_transition
        return gbm_transition(spot, variance, shocks, config, dt=dt, generator=generator)
    if config.model == "bates":
        from .bates import bates_transition
        return bates_transition(spot, variance, shocks, config, dt=dt, generator=generator)
    return heston_transition(spot, variance, shocks, config, dt=dt, generator=generator)


def market_shocks(config, rng, count=None, *, dt):
    """Seeded NumPy chance draws for scalar/batched search, never future paths.

    Heston's three-channel draw order is unchanged. Bates appends an actual
    Poisson jump count and a normal aggregate-size shock; search cannot drop
    jumps by silently invoking the Heston transition.
    """
    columns = [rng.normal(size=count), rng.random(size=count), rng.normal(size=count)]
    if config.model == "bates":
        columns.extend((rng.poisson(config.jump_intensity * dt, size=count),
                        rng.normal(size=count)))
    return np.stack(columns, axis=-1)


def heston_transition(spot, variance, shocks=None, config: HestonConfig | None = None, *, dt, generator=None):
    """One action-independent step; shocks[..., :] are normalV, uniformV, normalS.

    Andersen QE/QE-M, following QuantLib's HestonProcess.evolve. Explicit shocks
    permit common random numbers and native QuantLib parity. A local generator
    draws fresh conditional chance outcomes without changing the channel order.
    """
    config = config or HestonConfig()
    spot, variance = torch.broadcast_tensors(spot, variance)
    if shocks is None:
        zv = torch.randn(spot.shape, device=spot.device, dtype=spot.dtype, generator=generator)
        uv = torch.rand(spot.shape, device=spot.device, dtype=spot.dtype, generator=generator)
        zs = torch.randn(spot.shape, device=spot.device, dtype=spot.dtype, generator=generator)
    else:
        shocks = torch.as_tensor(shocks, dtype=spot.dtype, device=spot.device)
        zv, uv, zs = shocks.unbind(-1)
    decay = math.exp(-config.kappa * dt)
    mean = config.theta + (variance - config.theta) * decay
    variance_of_variance = (
        variance * config.sigma**2 * decay * (1 - decay) / config.kappa
        + config.theta * config.sigma**2 * (1 - decay)**2 / (2 * config.kappa)
    )
    epsilon = torch.finfo(spot.dtype).tiny
    psi = variance_of_variance / mean.square().clamp_min(epsilon)
    # Preserve the legacy QE switch; QE-M follows QuantLib's strict comparison.
    quadratic_branch = psi <= 1.5 if config.scheme == "qe" else psi < 1.5
    # Keep unused branches in their domains: torch.where evaluates both, and
    # an inactive sqrt/log singularity can otherwise poison pathwise gradients.
    quadratic_psi = torch.where(quadratic_branch, psi, torch.ones_like(psi))
    two_over_psi = 2 / quadratic_psi.clamp_min(epsilon)
    b2 = two_over_psi - 1 + two_over_psi.sqrt() * (two_over_psi - 1).clamp_min(0).sqrt()
    a = mean / (1 + b2)
    quadratic = a * (b2.clamp_min(0).sqrt() + zv).square()
    exponential_psi = torch.where(quadratic_branch, torch.full_like(psi, 2.), psi)
    probability_zero = (exponential_psi - 1) / (exponential_psi + 1)
    beta = (1 - probability_zero) / mean.clamp_min(epsilon)
    exponential = torch.log((1 - probability_zero) / (1 - uv).clamp_min(epsilon)) / beta
    point_mass = torch.where(uv > probability_zero, exponential, torch.zeros_like(exponential))
    next_variance = torch.where(quadratic_branch, quadratic, point_mass)
    common = 0.5 * dt * (config.kappa * config.rho / config.sigma - 0.5)
    k0 = -config.rho * config.kappa * config.theta * dt / config.sigma
    k1, k2 = common - config.rho / config.sigma, common + config.rho / config.sigma
    noise_weight = 0.5 * dt * (1 - config.rho**2)
    if config.scheme == "qe_m":
        # Integrate out both random inputs so E[S_next | S,v] = S at r=q=0.
        # Only the log-stock constant changes; variance and random draws do not.
        moment_coefficient = k2 + 0.5 * noise_weight
        active_a = torch.where(quadratic_branch, a, torch.zeros_like(a))
        quadratic_denominator = 1 - 2 * active_a * moment_coefficient
        exponential_denominator = torch.where(
            quadratic_branch, torch.ones_like(beta), beta - moment_coefficient)
        # The moment always exists when the coefficient is nonpositive. For
        # positive coefficients use QuantLib's domain condition, not clipping.
        if moment_coefficient > 0 and bool(((quadratic_denominator <= 0)
                                            | (exponential_denominator <= 0)).any()):
            raise ValueError("QE-M exponential moment does not exist; reduce the simulation time step")
        quadratic_log_moment = (moment_coefficient * b2 * active_a / quadratic_denominator
                                - 0.5 * torch.log(quadratic_denominator))
        active_probability = torch.where(quadratic_branch, torch.ones_like(probability_zero), probability_zero)
        exponential_log_moment = torch.log(
            active_probability + beta * (1 - active_probability) / exponential_denominator)
        k0 = (-torch.where(quadratic_branch, quadratic_log_moment, exponential_log_moment)
              - (k1 + 0.5 * noise_weight) * variance)
    noise_variance = noise_weight * (variance + next_variance)
    next_spot = spot * torch.exp(k0 + k1 * variance + k2 * next_variance
                                + noise_variance.clamp_min(0).sqrt() * zs)
    return next_spot, next_variance


def option_price(spot, variance, maturity, strike, market, *, kind="call"):
    """Price in float64 before restoring the input dtype; preserve gradients."""
    if kind not in ("call", "put"):
        raise ValueError("choose call or put")
    if not isinstance(spot, torch.Tensor):
        spot = torch.as_tensor(spot, dtype=torch.float64)
    reference = spot.to(torch.float64)
    call = call_price(reference, variance, maturity, strike, market)
    # A small put is the difference of large terms. Apply r=q=0 parity before
    # rounding the call price to the ledger dtype, not after it.
    value = call if kind == "call" else (call - reference + _tensor(strike, call)).clamp_min(0)
    return value.to(spot.dtype)


def variance_swap_price(variance, integrated_variance, remaining, market):
    """Conditional value of Bühler's variance leg, in variance-years units."""
    remaining = _tensor(remaining, variance).clamp_min(0)
    if market.model == "gbm":
        return integrated_variance + variance * remaining
    decay_integral = -torch.expm1(-market.kappa * remaining) / market.kappa
    return integrated_variance + market.theta * remaining + (variance - market.theta) * decay_integral


def mark_state(spot, variance, time_index, config: HedgingConfig, *, integrated_variance=None):
    """Current marks and signed liability value, independent of trading rules."""
    reference = spot.to(torch.float64)
    date_index = _tensor(time_index, reference)
    def price(contract):
        # Subtract integer dates first: expiry is exactly zero even when the
        # equivalent floating-point year fractions differ by a few ulps.
        remaining = (round(contract.maturity * config.time_grid.days_per_year)
                     - date_index * config.time_grid.step_days) / config.time_grid.days_per_year
        return contract.mark(reference, variance.to(torch.float64), remaining, config.market,
                             integrated_variance=integrated_variance).to(spot.dtype)
    marks = torch.stack([spot, *(price(contract) for contract in config.portfolio.hedges)], dim=-1)
    # Multiply in double precision before casting, as in the initial ledger.
    contract = config.portfolio.liability
    remaining = ((config.n_steps - date_index) * config.dt).clamp_min(0)
    value = contract.mark(reference, variance.to(torch.float64), remaining, config.market,
                          integrated_variance=integrated_variance)
    return marks, (config.portfolio.liability_quantity * value).to(spot.dtype)


@torch.no_grad()
def simulate_market_paths(market, time_grid: TimeGrid, n_paths: int, seed: int, *, device="cpu",
                          dtype=torch.float64, substeps=1, return_integrated_variance=False):
    """Simulate at finer internal steps, returning only unchanged trading dates.

    The original config and observation/calendar contract do not change.
    substeps=1 preserves the historical Heston RNG order exactly. Different
    resolutions are not implicitly Brownian-coupled by sharing a seed.
    """
    if n_paths < 1 or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("positive path count and integer simulation substeps required")
    generator = torch.Generator(device=device).manual_seed(seed)
    spot = torch.full((n_paths,), market.spot0, device=device, dtype=torch.float64)
    variance = torch.full_like(spot, market.v0)
    internal_dt = time_grid.dt / substeps
    spots, variances = [spot], [variance]
    integral = torch.zeros_like(variance)
    integrals = [integral]
    for _ in range(time_grid.n_steps):
        for _ in range(substeps):
            previous_variance = variance
            spot, variance = transition(spot, variance, config=market, dt=internal_dt, generator=generator)
            if return_integrated_variance:
                integral = integral + .5 * (previous_variance + variance) * internal_dt
        spots.append(spot)
        variances.append(variance)
        if return_integrated_variance:
            integrals.append(integral)
    spot, variance = torch.stack(spots, dim=1), torch.stack(variances, dim=1)
    result = (spot.to(dtype), variance.to(dtype))
    if return_integrated_variance:
        return (*result, torch.stack(integrals, dim=1).to(dtype))
    return result


@torch.no_grad()
def market_bank_from_paths(config, spot, variance, *, price_chunk_size=1024, dtype=None,
                           integrated_variance=None):
    """Price supplied trading-date paths with the shared model's marks."""
    if (spot.ndim != 2 or spot.shape != variance.shape or spot.shape[1] != config.n_steps + 1
            or len(spot) == 0 or price_chunk_size < 1):
        raise ValueError("nonempty [paths,n_steps+1] market states and positive pricing chunks required")
    n_paths, device = len(spot), spot.device
    dtype = dtype or spot.dtype
    times = torch.arange(config.n_steps + 1, device=device).expand(n_paths, -1).reshape(-1)
    flat_spot, flat_variance = spot.reshape(-1), variance.reshape(-1)
    if integrated_variance is not None and integrated_variance.shape != spot.shape:
        raise ValueError("integrated variance must follow the supplied market dates")
    flat_integral = None if integrated_variance is None else integrated_variance.reshape(-1)
    marks, liabilities = [], []
    for offset in range(0, flat_spot.numel(), price_chunk_size):
        sl = slice(offset, offset + price_chunk_size)
        mark, liability = mark_state(flat_spot[sl], flat_variance[sl], times[sl], config,
            integrated_variance=None if flat_integral is None else flat_integral[sl])
        marks.append(mark)
        liabilities.append(liability)
    return MarketBank(
        spot.to(dtype), variance.to(dtype),
        torch.cat(marks).reshape(n_paths, config.n_steps + 1, config.n_assets).to(dtype),
        torch.cat(liabilities).reshape(n_paths, config.n_steps + 1).to(dtype), config,
        None if integrated_variance is None else integrated_variance.to(dtype),
    )


@torch.no_grad()
def generate_market_bank(config, n_paths: int, seed: int, *, device="cpu", dtype=torch.float32,
                         price_chunk_size=1024, substeps=1):
    """Batched market paths and option marks; internal refinement preserves trading dates."""
    needs_integral = any(instrument.needs_integrated_variance for instrument in
                         (config.portfolio.liability, *config.portfolio.hedges))
    paths = simulate_market_paths(config.market, config.time_grid, n_paths, seed, device=device,
                                  substeps=substeps, return_integrated_variance=needs_integral)
    return market_bank_from_paths(config, *paths[:2], price_chunk_size=price_chunk_size, dtype=dtype,
                                  integrated_variance=paths[2] if needs_integral else None)


def _initial_state(premium, config):
    zero = torch.zeros_like(premium)
    cash = premium.clone() if config.portfolio.initial_cash is None else zero + config.portfolio.initial_cash
    positions = _tensor(config.portfolio.initial_positions or (0.,) * config.n_assets, premium)
    return LedgerState(cash, positions.expand(*premium.shape, config.n_assets).clone(), zero.clone(),
                       zero[..., None].expand(*premium.shape, config.n_assets).clone(), zero.clone())


def initial_ledger(reference: torch.Tensor, config: HedgingConfig):
    """Declared starting inventory/cash; model premium is the default cash."""
    contract = config.portfolio.liability
    # Match mark_state: price and apply quantity before casting to the ledger dtype.
    spot = _tensor(config.market.spot0, reference.to(torch.float64))
    premium = (config.portfolio.liability_quantity * contract.mark(
        spot, torch.zeros_like(spot) + config.market.v0, contract.maturity, config.market,
        integrated_variance=torch.zeros_like(spot))).to(reference.dtype)
    return _initial_state(torch.zeros_like(reference) + premium, config)


def initial_state(bank: MarketBank):
    return _initial_state(bank.liability[:, 0], bank.config)


def map_action(raw, config: HedgingConfig):
    lower = _tensor(config.execution.vector("holding_lower", config.n_assets), raw)
    upper = _tensor(config.execution.vector("holding_upper", config.n_assets), raw)
    finite = torch.isfinite(lower) & torch.isfinite(upper)
    lo, hi = torch.where(finite, lower, 0.), torch.where(finite, upper, 0.)
    bounded = lo + (hi - lo) * (torch.tanh(raw) + 1) / 2
    return torch.where(finite, bounded, raw).clamp(min=lower, max=upper)


def transaction_cost(trade, marks, config: HedgingConfig):
    """Commission floor on nonzero trades, plus quadratic cost and fixed ticket.

    Minimum commission replaces a smaller proportional fee; it is not an
    additional fee. HOLD costs zero, and mandatory liquidation pays all fees.
    """
    commission = trade.abs() * (marks.abs() * _tensor(config.execution.proportional, marks)
                               + _tensor(config.execution.per_unit, marks))
    if any(config.execution.vector("minimum_commission", config.n_assets)):
        commission = torch.where(trade != 0,
                                 torch.maximum(commission, _tensor(config.execution.minimum_commission, marks)),
                                 torch.zeros_like(commission))
    if any(config.execution.vector("commission_cap", config.n_assets)):
        cap = _tensor(config.execution.commission_cap, marks)
        commission = torch.where(cap > 0, torch.minimum(commission, cap), commission)
    return (commission
            + trade.square() * marks.abs() * _tensor(config.execution.quadratic, marks)
            + (trade != 0).to(marks.dtype) * _tensor(config.execution.fixed_ticket, marks)).sum(-1)


def feasible_targets(previous, targets, config: HedgingConfig, *, liquidating=False):
    """Per-instrument legality, broadcasting over paths and candidate actions.

    Limits apply to trade increments, not desired target magnitudes. HOLD is
    always allowed within holding bounds. Mandatory terminal closeout waives
    the minimum-order size only; contract lots and all trading fees still apply.
    The tolerance covers floating-point representation, not economic slack.
    """
    previous, targets = torch.broadcast_tensors(previous, targets)
    lower = _tensor(config.execution.vector("holding_lower", config.n_assets), targets)
    upper = _tensor(config.execution.vector("holding_upper", config.n_assets), targets)
    valid = torch.isfinite(targets) & (targets >= lower) & (targets <= upper)
    if not any(config.execution.vector("minimum_trade", config.n_assets) + config.execution.vector("trade_lot", config.n_assets)):
        return valid
    trade = targets - previous
    hold = trade == 0
    lot = _tensor(config.execution.trade_lot, targets)
    scale = torch.maximum(torch.maximum(targets.abs(), previous.abs()), lot)
    tolerance = 8 * torch.finfo(targets.dtype).eps * scale
    if not liquidating:
        valid = valid & (hold | (trade.abs() + tolerance >= _tensor(config.execution.minimum_trade, targets)))
    if any(config.execution.vector("trade_lot", config.n_assets)):
        units = (trade / torch.where(lot > 0, lot, torch.ones_like(lot))).round()
        valid = valid & ((lot == 0) | hold | ((units != 0) & ((trade - units * lot).abs() <= tolerance)))
    return valid


def trade_step(state: LedgerState, target_positions, marks, config: HedgingConfig, *, liquidating=False):
    """Execute feasible targets without projection or observation-dependent rules."""
    if not bool(feasible_targets(
            state.positions, target_positions, config, liquidating=liquidating).all()):
        raise ValueError("target holdings violate holding bounds, minimum-trade and lot rules")
    trade = target_positions - state.positions
    costs = (torch.zeros_like(state.cash) if liquidating and not config.settlement.charge_liquidation_costs
             else transaction_cost(trade, marks, config))
    return LedgerState(
        state.cash - (trade * marks).sum(-1) - costs, target_positions,
        state.total_cost + costs, state.turnover + trade.abs(),
        state.tickets + (trade != 0).sum(-1),
    )


def settle_ledger(state: LedgerState, terminal_marks, config: HedgingConfig):
    """Terminal cash equivalent; marking inventory creates no fictitious trade."""
    if config.settlement.mode == "mark_to_market":
        return LedgerState(state.cash + (state.positions * terminal_marks).sum(-1),
                           torch.zeros_like(state.positions), state.total_cost,
                           state.turnover, state.tickets)
    return trade_step(state, torch.zeros_like(state.positions), terminal_marks, config, liquidating=True)


def liquidate(state: LedgerState, terminal_marks, terminal_payoff, config: HedgingConfig):
    """Settle the configured episode; the legacy function name is retained."""
    liquidation_value = (state.positions * terminal_marks).sum(-1)
    final = settle_ledger(state, terminal_marks, config)
    liquidation_cost = final.total_cost - state.total_cost
    pnl = final.cash - terminal_payoff
    return dict(terminal_loss=-pnl, terminal_pnl=pnl, transaction_cost=final.total_cost,
                total_cost=final.total_cost,
                turnover=final.turnover, tickets=final.tickets,
                cash_before_liquidation=state.cash, liquidation_value=liquidation_value,
                liquidation_cost=liquidation_cost, liability_payoff=terminal_payoff)


def terminal_loss(bank: MarketBank, state: LedgerState):
    return liquidate(state, bank.marks[:, -1], bank.liability[:, -1], bank.config)["terminal_loss"]


def ledger_from_positions(bank: MarketBank, positions):
    if positions.shape != (bank.spot.shape[0], bank.config.n_decisions, bank.config.n_assets):
        raise ValueError("positions must have shape [paths, n_decisions, n_assets]")
    state = initial_state(bank)
    initial_positions = state.positions
    for t in range(bank.config.n_decisions):
        state = trade_step(state, positions[:, t], bank.marks[:, t], bank.config)
    result = liquidate(state, bank.marks[:, -1], bank.liability[:, -1], bank.config)
    previous = torch.cat((initial_positions[:, None], positions[:, :-1]), dim=1)
    result["constraint_violations"] = (~feasible_targets(previous, positions, bank.config)).sum((-1, -2))
    return result


def numpy_ledger(marks, positions, initial_cash, liability_payoff, config: HedgingConfig):
    """Independent sequential cash reconstruction, including the final sell-out.

    Deliberately uses scalar instrument loops and no Torch ledger helper.
    """
    marks, positions = np.asarray(marks), np.asarray(positions)
    execution = {name: config.execution.vector(name, config.n_assets) for name in EXECUTION_FIELDS}
    cash = np.broadcast_to(np.asarray(initial_cash), (len(marks),)).astype(np.float64).copy()
    previous = np.broadcast_to(config.portfolio.initial_positions or (0.,) * config.n_assets,
                               (len(marks), config.n_assets)).copy()
    costs, tickets = np.zeros(len(marks)), np.zeros(len(marks))
    turnover = np.zeros((len(marks), config.n_assets))
    for t in range(positions.shape[1] + 1):
        prices = marks[:, min(t, marks.shape[1]-1)]
        if t == positions.shape[1] and config.settlement.mode == "mark_to_market":
            cash += (previous * prices).sum(-1)
            break
        target = positions[:, t] if t < positions.shape[1] else np.zeros_like(previous)
        for j in range(config.n_assets):
            trade = target[:, j] - previous[:, j]
            commission = np.abs(trade) * (execution["proportional"][j] * np.abs(prices[:, j]) + execution["per_unit"][j])
            if execution["minimum_commission"][j]:
                commission = np.where(trade != 0, np.maximum(commission, execution["minimum_commission"][j]), 0)
            if execution["commission_cap"][j]:
                commission = np.minimum(commission, execution["commission_cap"][j])
            fee = (commission
                   + execution["quadratic"][j] * trade**2 * np.abs(prices[:, j])
                   + execution["fixed_ticket"][j] * (trade != 0))
            if t == positions.shape[1] and not config.settlement.charge_liquidation_costs:
                fee = np.zeros_like(fee)
            cash -= trade * prices[:, j] + fee
            costs += fee
            turnover[:, j] += np.abs(trade)
            tickets += trade != 0
        previous = target.copy()
    pnl = cash - np.asarray(liability_payoff)
    return dict(terminal_loss=-pnl, terminal_pnl=pnl, transaction_cost=costs,
                turnover=turnover, tickets=tickets)


def market_parameter_names(config):
    if config.model == "gbm":
        return ("mu",)
    names = ("kappa", "theta", "sigma", "rho")
    if config.model == "bates":
        names += ("jump_intensity", "jump_mean", "jump_std")
    return names


def instrument_names(config: HedgingConfig):
    return ("stock", *(f"{option.kind}_{index + 1}" for index, option in enumerate(config.portfolio.hedges)))


def _execution_observation_fields(config):
    # Keep existing checkpoints' basic observation schema. New fee components
    # add named fields when selected; never encode infinity in network inputs.
    return EXECUTION_FIELDS[:8] + tuple(name for name in EXECUTION_FIELDS[8:]
        if any(config.execution.vector(name, config.n_assets)))


def _contract_observation_parameters(config):
    book = config.portfolio
    values = {f"liability_{key}": value for key, value in book.liability.features.items()}
    # Preserve the original option schema's ordering.
    values = {key: value for key, value in values.items() if key != "liability_kind"}
    values["liability_quantity"] = book.liability_quantity
    if "kind" in book.liability.features:
        values["liability_kind"] = book.liability.features["kind"]
    keys = dict.fromkeys(key for hedge in book.hedges for key in hedge.features)
    for key in keys:
        for i, hedge in enumerate(book.hedges):
            if key in hedge.features:
                values[f"hedge_{key}_{i+1}"] = hedge.features[key]
    return values


def observation_fields(config: HedgingConfig):
    instruments = instrument_names(config)
    return (
        "time_fraction", "spot", "variance", "cash",
        *(f"{name}_position" for name in instruments),
        *(f"{name}_mid" for name in instruments),
        *(f"{field}_{name}" for field in _execution_observation_fields(config) for name in instruments),
        *(f"{side}_unbounded" for side in ("holding_lower", "holding_upper")
          if getattr(config.execution, side) is None),
        *market_parameter_names(config.market),
        *_contract_observation_parameters(config),
        "dt", "spot0", "v0",
    )


def decode_market_observation(observed, config: HedgingConfig):
    """Inverse of the shared market-state normalization, using named fields."""
    fields = observation_fields(config)
    return (observed[..., fields.index("spot")] * config.market.spot0,
            observed[..., fields.index("variance")] * max(config.market.v0, 1e-4))


def observation_from_state(spot, variance, time_index, state: LedgerState, marks, config: HedgingConfig):
    """Causal state; all execution fields are present even when their value is zero.

    Contract kinds use +1 for calls and -1 for puts. Values use S0 and v0 scales
    where appropriate. A method needing a global ES threshold supplies it itself.
    """
    market, portfolio = config.market, config.portfolio
    time = torch.zeros_like(spot) + _tensor(time_index, spot) / config.n_steps
    values = [time[..., None], (spot / market.spot0)[..., None],
              (variance / max(market.v0, 1e-4))[..., None], (state.cash / market.spot0)[..., None],
              state.positions, marks / market.spot0]
    for name in _execution_observation_fields(config):
        vector = config.execution.vector(name, config.n_assets)
        if getattr(config.execution, name) is None:
            vector = (0.,) * config.n_assets
        values.append(_tensor(vector, spot).expand(*spot.shape, config.n_assets))
    for side in ("holding_lower", "holding_upper"):
        if getattr(config.execution, side) is None:
            values.append(torch.ones_like(spot)[..., None])
    market_parameters = tuple(getattr(market, name) for name in market_parameter_names(market))
    constants = (*market_parameters, *_contract_observation_parameters(config).values(),
                 config.dt, market.spot0, market.v0)
    values.append(_tensor(constants, spot).expand(*spot.shape, len(constants)))
    return torch.cat(values, dim=-1)


def observation(bank: MarketBank, time_index: int, state: LedgerState):
    return observation_from_state(bank.spot[:, time_index], bank.variance[:, time_index],
                                  time_index, state, bank.marks[:, time_index], bank.config)
