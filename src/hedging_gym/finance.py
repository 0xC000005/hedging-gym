"""Batched market prices and paths with one differentiable portfolio cash ledger.

Heston uses a quadratic-exponential variance step and an approximate log-spot
step (not an exact martingale correction). GBM is exact at trading dates; Bates
adds compensated independent jumps. Independent references live in tests.
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


BANK_FIELDS = ("spot", "variance", "marks", "liability")


def bank_to(bank: MarketBank, device) -> MarketBank:
    return MarketBank(*(getattr(bank, key).to(device) for key in BANK_FIELDS), bank.config)


def bank_subset(bank: MarketBank, indices) -> MarketBank:
    return MarketBank(*(getattr(bank, key)[indices] for key in BANK_FIELDS), bank.config)


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

    Use a 48-node [0,150] panel and append panels for short or
    low-variance states. Positive-intensity Bates uses 96 nodes per panel:
    large jumps create far-moneyness oscillations unresolved by 48 nodes.
    A sole panel has material one-day errors.
    The cutoff is at least 12 / sqrt(E[integrated variance]); it is a numerical
    rule qualified by independent QuantLib checks, not a universal error bound.
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
    panels = torch.ceil(12.0 / (150.0 * expected_variance.clamp_min(1e-20).sqrt()))
    # Bucket nearby cutoffs together, without ever reducing the requested cutoff.
    buckets = torch.ceil(torch.log2(panels.clamp_min(1))).to(torch.int64)
    if bool((buckets[active] > 10).any()):
        raise ValueError("state needs more than 1024 Fourier panels; requalify its pricing domain")
    order = 96 if config.model == "bates" and config.jump_intensity > 0 else 48
    nodes_np, weights_np = _quadrature(order)
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

    Explicit shocks permit common random numbers and independent NumPy parity.
    An optional local generator draws fresh conditional chance outcomes.
    """
    config = config or HestonConfig()
    spot, variance = torch.broadcast_tensors(spot, variance)
    if shocks is None:
        zv = torch.randn(spot.shape, device=spot.device, dtype=spot.dtype, generator=generator)
        uv = torch.rand(spot.shape, device=spot.device, dtype=spot.dtype, generator=generator)
        zs = torch.randn(spot.shape, device=spot.device, dtype=spot.dtype, generator=generator)
    else:
        zv, uv, zs = shocks.unbind(-1)
    decay = math.exp(-config.kappa * dt)
    mean = config.theta + (variance - config.theta) * decay
    variance_of_variance = (
        variance * config.sigma**2 * decay * (1 - decay) / config.kappa
        + config.theta * config.sigma**2 * (1 - decay)**2 / (2 * config.kappa)
    )
    epsilon = torch.finfo(spot.dtype).tiny
    psi = variance_of_variance / mean.square().clamp_min(epsilon)
    two_over_psi = 2 / psi.clamp_min(epsilon)
    b2 = two_over_psi - 1 + two_over_psi.sqrt() * (two_over_psi - 1).clamp_min(0).sqrt()
    quadratic = mean / (1 + b2) * (b2.clamp_min(0).sqrt() + zv).square()
    probability_zero = (psi - 1) / (psi + 1)
    beta = (1 - probability_zero) / mean.clamp_min(epsilon)
    exponential = torch.log((1 - probability_zero) / (1 - uv).clamp_min(epsilon)) / beta
    point_mass = torch.where(uv > probability_zero, exponential, torch.zeros_like(exponential))
    next_variance = torch.where(psi <= 1.5, quadratic, point_mass)
    common = 0.5 * dt * (config.kappa * config.rho / config.sigma - 0.5)
    k0 = -config.rho * config.kappa * config.theta * dt / config.sigma
    k1, k2 = common - config.rho / config.sigma, common + config.rho / config.sigma
    noise_variance = 0.5 * dt * (1 - config.rho**2) * (variance + next_variance)
    next_spot = spot * torch.exp(k0 + k1 * variance + k2 * next_variance
                                + noise_variance.clamp_min(0).sqrt() * zs)
    return next_spot, next_variance


def option_price(spot, variance, maturity, strike, market, *, kind="call"):
    """Price a European call or put; r=q=0 put-call parity preserves gradients."""
    if kind not in ("call", "put"):
        raise ValueError("choose call or put")
    call = call_price(spot, variance, maturity, strike, market)
    return call if kind == "call" else (call - _tensor(spot, call) + _tensor(strike, call)).clamp_min(0)


def mark_state(spot, variance, time_index, config: HedgingConfig):
    """Current marks and signed liability value, independent of trading rules."""
    reference = spot.to(torch.float64)
    date_index = _tensor(time_index, reference)
    contracts = config.portfolio.hedges
    if contracts:
        strikes = _tensor([option.strike for option in contracts], reference)
        # Contract dates are validated on the selected clock. Subtract dates
        # before converting to years so expiry is exactly zero, not a tiny
        # positive maturity that would trigger needless Fourier refinement.
        expiry_dates = [round(option.maturity * config.time_grid.days_per_year) for option in contracts]
        maturity = (_tensor(expiry_dates, reference) - date_index[..., None]) * config.dt
        calls = call_price(reference[..., None], variance[..., None], maturity, strikes, config.market)
        puts = (calls - reference[..., None] + strikes).clamp_min(0)
        is_put = torch.tensor([option.kind == "put" for option in contracts], device=spot.device)
        prices = torch.where(is_put, puts, calls).to(spot.dtype)
        marks = torch.cat((spot[..., None], prices), dim=-1)
    else:
        marks = spot[..., None]
    liability = config.portfolio.liability
    value = option_price(reference, variance, ((config.n_steps - date_index) * config.dt).clamp_min(0),
                         liability.strike, config.market, kind=liability.kind)
    return marks, (config.portfolio.liability_quantity * value).to(spot.dtype)


@torch.no_grad()
def simulate_market_paths(market, time_grid: TimeGrid, n_paths: int, seed: int, *, device="cpu",
                          dtype=torch.float64, substeps=1):
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
    for _ in range(time_grid.n_steps):
        for _ in range(substeps):
            spot, variance = transition(spot, variance, config=market, dt=internal_dt, generator=generator)
        spots.append(spot)
        variances.append(variance)
    spot, variance = torch.stack(spots, dim=1), torch.stack(variances, dim=1)
    return spot.to(dtype), variance.to(dtype)


@torch.no_grad()
def market_bank_from_paths(config, spot, variance, *, price_chunk_size=1024, dtype=None):
    """Price supplied trading-date paths with the shared model's marks."""
    if (spot.ndim != 2 or spot.shape != variance.shape or spot.shape[1] != config.n_steps + 1
            or len(spot) == 0 or price_chunk_size < 1):
        raise ValueError("nonempty [paths,n_steps+1] market states and positive pricing chunks required")
    n_paths, device = len(spot), spot.device
    dtype = dtype or spot.dtype
    times = torch.arange(config.n_steps + 1, device=device).expand(n_paths, -1).reshape(-1)
    flat_spot, flat_variance = spot.reshape(-1), variance.reshape(-1)
    marks, liabilities = [], []
    for offset in range(0, flat_spot.numel(), price_chunk_size):
        sl = slice(offset, offset + price_chunk_size)
        mark, liability = mark_state(flat_spot[sl], flat_variance[sl], times[sl], config)
        marks.append(mark)
        liabilities.append(liability)
    return MarketBank(
        spot.to(dtype), variance.to(dtype),
        torch.cat(marks).reshape(n_paths, config.n_steps + 1, config.n_assets).to(dtype),
        torch.cat(liabilities).reshape(n_paths, config.n_steps + 1).to(dtype), config,
    )


@torch.no_grad()
def generate_market_bank(config, n_paths: int, seed: int, *, device="cpu", dtype=torch.float32,
                         price_chunk_size=1024, substeps=1):
    """Batched market paths and option marks; internal refinement preserves trading dates."""
    spot, variance = simulate_market_paths(config.market, config.time_grid, n_paths, seed, device=device, substeps=substeps)
    return market_bank_from_paths(config, spot, variance, price_chunk_size=price_chunk_size, dtype=dtype)


def initial_ledger(reference: torch.Tensor, config: HedgingConfig):
    """Zero initial holdings and the identical authoritative liability premium."""
    contract = config.portfolio.liability
    premium = config.portfolio.liability_quantity * option_price(
        _tensor(config.market.spot0, reference), config.market.v0, contract.maturity,
        contract.strike, config.market, kind=contract.kind)
    zero = torch.zeros_like(reference)
    return LedgerState(zero + premium, zero[..., None].expand(*zero.shape, config.n_assets).clone(),
                       zero.clone(), zero[..., None].expand(*zero.shape, config.n_assets).clone(), zero.clone())


def initial_state(bank: MarketBank):
    zero = torch.zeros_like(bank.spot[:, 0])
    return LedgerState(bank.liability[:, 0].clone(), bank.marks[:, 0].new_zeros((len(zero), bank.config.n_assets)),
                       zero.clone(), bank.marks[:, 0].new_zeros((len(zero), bank.config.n_assets)), zero.clone())


def map_action(raw, config: HedgingConfig):
    lower, upper = _tensor(config.execution.holding_lower, raw), _tensor(config.execution.holding_upper, raw)
    return lower + (upper - lower) * (torch.tanh(raw) + 1) / 2


def hybrid_action(modes, sizes, current_positions, config: HedgingConfig):
    """Hold=0, buy=1, sell=2; sizes are legal fractions in [0,1]."""
    lower, upper = _tensor(config.execution.holding_lower, sizes), _tensor(config.execution.holding_upper, sizes)
    if bool(((sizes < 0) | (sizes > 1)).any()) or bool(((modes < 0) | (modes > 2)).any()):
        raise ValueError("hybrid action requires legal modes and unit interval sizes")
    return torch.where(modes == 1, current_positions + sizes * (upper - current_positions),
                       torch.where(modes == 2, current_positions - sizes * (current_positions - lower),
                                   current_positions))


def transaction_cost(trade, marks, config: HedgingConfig):
    """Commission floor on nonzero trades, plus quadratic cost and fixed ticket.

    Minimum commission replaces a smaller proportional fee; it is not an
    additional fee. HOLD costs zero, and mandatory liquidation pays all fees.
    """
    commission = trade.abs() * marks * _tensor(config.execution.proportional, marks)
    if any(config.execution.vector("minimum_commission", config.n_assets)):
        commission = torch.where(trade != 0,
                                 torch.maximum(commission, _tensor(config.execution.minimum_commission, marks)),
                                 torch.zeros_like(commission))
    return (commission
            + trade.square() * marks * _tensor(config.execution.quadratic, marks)
            + (trade != 0).to(marks.dtype) * _tensor(config.execution.fixed_ticket, marks)).sum(-1)


def feasible_targets(previous, targets, config: HedgingConfig, *, liquidating=False):
    """Per-instrument legality, broadcasting over paths and candidate actions.

    Limits apply to trade increments, not desired target magnitudes. HOLD is
    always allowed within holding bounds. Mandatory terminal closeout waives
    the minimum-order size only; contract lots and all trading fees still apply.
    The tolerance covers floating-point representation, not economic slack.
    """
    previous, targets = torch.broadcast_tensors(previous, targets)
    lower, upper = _tensor(config.execution.holding_lower, targets), _tensor(config.execution.holding_upper, targets)
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
    costs = transaction_cost(trade, marks, config)
    return LedgerState(
        state.cash - (trade * marks).sum(-1) - costs, target_positions,
        state.total_cost + costs, state.turnover + trade.abs(),
        state.tickets + (trade != 0).sum(-1),
    )


def liquidate(state: LedgerState, terminal_marks, terminal_payoff, config: HedgingConfig):
    liquidation_value = (state.positions * terminal_marks).sum(-1)
    liquidation_cost = transaction_cost(-state.positions, terminal_marks, config)
    final = trade_step(state, torch.zeros_like(state.positions), terminal_marks, config, liquidating=True)
    pnl = final.cash - terminal_payoff
    return dict(terminal_loss=-pnl, terminal_pnl=pnl, transaction_cost=final.total_cost,
                total_cost=final.total_cost,
                turnover=final.turnover, tickets=final.tickets,
                cash_before_liquidation=state.cash, liquidation_value=liquidation_value,
                liquidation_cost=liquidation_cost, liability_payoff=terminal_payoff)


def terminal_loss(bank: MarketBank, state: LedgerState):
    return liquidate(state, bank.marks[:, -1], bank.liability[:, -1], bank.config)["terminal_loss"]


def ledger_from_positions(bank: MarketBank, positions):
    if positions.shape != (bank.spot.shape[0], bank.config.n_steps, bank.config.n_assets):
        raise ValueError("positions must have shape [paths, n_steps, n_assets]")
    state = initial_state(bank)
    for t in range(bank.config.n_steps):
        state = trade_step(state, positions[:, t], bank.marks[:, t], bank.config)
    result = liquidate(state, bank.marks[:, -1], bank.liability[:, -1], bank.config)
    previous = torch.cat((torch.zeros_like(positions[:, :1]), positions[:, :-1]), dim=1)
    result["constraint_violations"] = (~feasible_targets(previous, positions, bank.config)).sum((-1, -2))
    return result


def numpy_ledger(marks, positions, initial_cash, liability_payoff, config: HedgingConfig):
    """Independent sequential cash reconstruction, including the final sell-out.

    Deliberately uses scalar instrument loops and no Torch ledger helper.
    """
    marks, positions = np.asarray(marks), np.asarray(positions)
    execution = {name: config.execution.vector(name, config.n_assets) for name in EXECUTION_FIELDS}
    cash = np.broadcast_to(np.asarray(initial_cash), (len(marks),)).astype(np.float64).copy()
    previous = np.zeros((len(marks), config.n_assets))
    costs, tickets = np.zeros(len(marks)), np.zeros(len(marks))
    turnover = np.zeros((len(marks), config.n_assets))
    for t in range(marks.shape[1]):
        target = positions[:, t] if t < positions.shape[1] else np.zeros_like(previous)
        for j in range(config.n_assets):
            trade = target[:, j] - previous[:, j]
            commission = execution["proportional"][j] * np.abs(trade) * marks[:, t, j]
            if execution["minimum_commission"][j]:
                commission = np.where(trade != 0, np.maximum(commission, execution["minimum_commission"][j]), 0)
            fee = (commission
                   + execution["quadratic"][j] * trade**2 * marks[:, t, j]
                   + execution["fixed_ticket"][j] * (trade != 0))
            cash -= trade * marks[:, t, j] + fee
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


def observation_fields(config: HedgingConfig):
    instruments = instrument_names(config)
    return (
        "time_fraction", "spot", "variance", "cash",
        *(f"{name}_position" for name in instruments),
        *(f"{name}_mid" for name in instruments),
        *(f"{field}_{name}" for field in EXECUTION_FIELDS for name in instruments),
        *market_parameter_names(config.market),
        "liability_strike", "liability_maturity", "liability_quantity", "liability_kind",
        *(f"hedge_strike_{i+1}" for i in range(len(config.portfolio.hedges))),
        *(f"hedge_maturity_{i+1}" for i in range(len(config.portfolio.hedges))),
        *(f"hedge_kind_{i+1}" for i in range(len(config.portfolio.hedges))),
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
    for name in EXECUTION_FIELDS:
        values.append(_tensor(config.execution.vector(name, config.n_assets), spot).expand(*spot.shape, config.n_assets))
    market_parameters = tuple(getattr(market, name) for name in market_parameter_names(market))
    liability = portfolio.liability
    constants = (*market_parameters, liability.strike, liability.maturity,
                 portfolio.liability_quantity, 1.0 if liability.kind == "call" else -1.0,
                 *(option.strike for option in portfolio.hedges),
                 *(option.maturity for option in portfolio.hedges),
                 *(1.0 if option.kind == "call" else -1.0 for option in portfolio.hedges),
                 config.dt, market.spot0, market.v0)
    values.append(_tensor(constants, spot).expand(*spot.shape, len(constants)))
    return torch.cat(values, dim=-1)


def observation(bank: MarketBank, time_index: int, state: LedgerState):
    return observation_from_state(bank.spot[:, time_index], bank.variance[:, time_index],
                                  time_index, state, bank.marks[:, time_index], bank.config)
