"""ADAPTATION: batched market backends and one differentiable cash ledger.

The characteristic function and 48-node Gauss--Legendre panels come from
``finance_first_v1.benchmark``; the transition is the PFHedge-style Andersen
quadratic-exponential/log-spot scheme in ``gpc_transfer_v1.finance.heston_qem``.
The latter is a time discretization, not an exact Heston transition. In
particular its legacy name does not establish exact martingale correction.
Heston remains the historical default. GBM and Bates share the same book;
their pricing/transition implementations have separate QuantLib references.
The cash ledger currently requires r=q=0.

There are 30 trading intervals of 1/252 year, decisions at 0,...,29, and
liability settlement at 30/252. The common benchmark uses one hedge call at
60/252; low-level config defaults retain the historical two-call book at
60/252 and 90/252. Hedge calls are sold at their selected-model marks at
settlement. Every nonzero instrument
trade, including liquidation, pays its actual fixed ticket. A ticket's hard
activation has no pathwise gradient; discrete estimators belong to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np
import torch

EXECUTION_FIELDS = ("proportional", "quadratic", "fixed_ticket", "minimum_commission",
                    "minimum_trade", "trade_lot", "holding_lower", "holding_upper")


@lru_cache(maxsize=None)
def _quadrature(order: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Retained Fourier quadrature, now independent of the historical experiment."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    upper = 150.0
    return (nodes + 1.0) * upper / 2.0, weights * upper / 2.0


@dataclass(frozen=True)
class HedgeBookConfig:
    """Instrument, calendar and execution contract shared by market models."""
    n_steps: int = 30
    dt: float = 1.0 / 252.0
    spot0: float = 1.0
    v0: float = 0.04
    r: float = 0.0
    q: float = 0.0
    liability_strike: float = 1.0
    hedge_strikes: tuple[float, ...] = (0.95, 1.05)
    hedge_maturities: tuple[float, ...] = (60.0 / 252.0, 90.0 / 252.0)
    holding_lower: tuple[float, ...] = (-1.0, -1.0, -1.0)
    holding_upper: tuple[float, ...] = (2.0, 1.0, 1.0)
    proportional: tuple[float, ...] = (0.0005, 0.01, 0.01)
    quadratic: tuple[float, ...] = (0.0, 0.0, 0.0)
    fixed_ticket: tuple[float, ...] = (0.0, 0.0, 0.0)
    minimum_commission: tuple[float, ...] | None = None
    minimum_trade: tuple[float, ...] | None = None
    trade_lot: tuple[float, ...] | None = None
    execution_features: bool = False

    @property
    def n_assets(self):
        return 1 + len(self.hedge_strikes)

    def __post_init__(self):
        for name in ("hedge_strikes", "hedge_maturities", "holding_lower", "holding_upper",
                     "proportional", "quadratic", "fixed_ticket"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("minimum_commission", "minimum_trade", "trade_lot"):
            values = getattr(self, name)
            object.__setattr__(self, name, (0.0,) * self.n_assets if values is None else tuple(values))
        # Legacy checkpoints keep their input shape. New execution rules may not
        # be hidden from a controller; benchmark presets expose even zero rules.
        if any(self.minimum_commission + self.minimum_trade + self.trade_lot):
            object.__setattr__(self, "execution_features", True)
        if not all(math.isfinite(x) for x in (self.dt, self.spot0, self.v0,
                                              self.liability_strike, *self.hedge_strikes,
                                              *self.hedge_maturities)):
            raise ValueError("initial state and contract terms must be finite")
        if self.r != 0.0 or self.q != 0.0:
            raise ValueError("the shared cash ledger currently implements r=q=0")
        if self.n_steps < 1 or self.dt <= 0 or self.spot0 <= 0 or self.v0 < 0:
            raise ValueError("invalid horizon or initial market state")
        if not self.hedge_strikes or len(self.hedge_strikes) != len(self.hedge_maturities):
            raise ValueError("each hedge call needs a strike and maturity")
        if min(self.hedge_strikes + (self.liability_strike,)) <= 0:
            raise ValueError("strikes must be positive")
        if min(self.hedge_maturities) <= self.n_steps * self.dt:
            raise ValueError("hedge calls must outlive the liability")
        for name in EXECUTION_FIELDS:
            values = getattr(self, name)
            if len(values) != self.n_assets or not all(math.isfinite(x) for x in values):
                raise ValueError("bounds and costs need one finite value per instrument")
        if any(lo > 0 or hi < 0 or lo >= hi for lo, hi in
               zip(self.holding_lower, self.holding_upper)):
            raise ValueError("holding bounds must contain the zero initial portfolio")
        if min(self.proportional + self.quadratic + self.fixed_ticket
               + self.minimum_commission + self.minimum_trade + self.trade_lot) < 0:
            raise ValueError("trading costs and execution limits must be nonnegative")


@dataclass(frozen=True)
class HestonConfig(HedgeBookConfig):
    kappa: float = 3.0
    theta: float = 0.04
    sigma: float = 0.3
    rho: float = -0.5
    model: str = "heston"

    def __post_init__(self):
        super().__post_init__()
        if (self.model not in ("heston", "bates") or not all(math.isfinite(x) for x in
                (self.kappa, self.theta, self.sigma, self.rho))
                or min(self.kappa, self.theta, self.sigma) <= 0 or not -1 <= self.rho <= 1):
            raise ValueError("invalid Heston coefficients")


@dataclass(frozen=True)
class GBMConfig(HedgeBookConfig):
    """Constant variance v0; mu is physical drift, not the pricing drift."""
    mu: float = 0.0
    model: str = "gbm"

    def __post_init__(self):
        super().__post_init__()
        if self.model != "gbm" or not math.isfinite(self.mu) or self.v0 <= 0:
            raise ValueError("GBM requires finite drift and positive variance")


@dataclass(frozen=True)
class BatesConfig(HestonConfig):
    """Heston plus independent compensated compound-Poisson stock jumps.

    Jump log sizes are normal(jump_mean,jump_std squared). These initial
    development coefficients are explicit, not calibrated market evidence.
    """
    jump_intensity: float = 1.0
    jump_mean: float = -0.1
    jump_std: float = 0.2
    model: str = "bates"

    def __post_init__(self):
        super().__post_init__()
        if (self.model != "bates" or not all(math.isfinite(x) for x in
                (self.jump_intensity, self.jump_mean, self.jump_std))
                or min(self.jump_intensity, self.jump_std) < 0):
            raise ValueError("invalid Bates jump coefficients")


def config_from_dict(values):
    """Load named models while preserving historical Heston checkpoint configs."""
    model = values.get("model", "heston")
    if model == "heston":
        return HestonConfig(**values)
    if model == "gbm":
        return GBMConfig(**values)
    if model == "bates":
        return BatesConfig(**values)
    raise ValueError(f"unknown market model: {model}")


def common_config(*, model="heston", **changes):
    """Approved simple benchmark: stock, one ATM 60-day call and cash.

    HestonConfig() retains the historical two-call book for old results.
    Keyword changes define disclosed development cost/market configurations.
    """
    values = dict(hedge_strikes=(1.0,), hedge_maturities=(60.0 / 252.0,),
                  holding_lower=(-1.0, -1.0), holding_upper=(2.0, 1.0),
                  proportional=(0.0005, 0.01), quadratic=(0.0, 0.0),
                  fixed_ticket=(0.0, 0.0))
    values.update(changes)
    return config_from_dict(dict(values, model=model))


@dataclass
class MarketBank:
    spot: torch.Tensor                 # [paths, n_steps + 1]
    variance: torch.Tensor
    marks: torch.Tensor                # [paths, n_steps + 1, n_assets]
    liability: torch.Tensor            # positive value of one owed call
    config: HedgeBookConfig


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

    Reuse the legacy 48-node [0,150] panel and append panels for short or
    low-variance states. Positive-intensity Bates uses 96 nodes per panel:
    large jumps create far-moneyness oscillations unresolved by 48 nodes.
    A sole legacy panel has material one-day errors.
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


def quantlib_call_price(spot, variance, maturity, strike, config):
    """Independent QuantLib pricing reference on the exact 1/252-year grid.

    NullCalendar counts every artificial trading date under Business252.
    Reject off-grid times rather than silently rounding the financial horizon.
    """
    import QuantLib as ql

    if maturity == 0:
        return max(float(spot) - float(strike), 0.0)
    days = round(float(maturity) * 252)
    if days < 1 or not math.isclose(days / 252, float(maturity), abs_tol=1e-12):
        raise ValueError("QuantLib oracle requires exact positive n/252 maturity")
    evaluation = ql.Date(2, ql.January, 2024)
    previous_evaluation = ql.Settings.instance().evaluationDate
    try:
        ql.Settings.instance().evaluationDate = evaluation
        day_count = ql.Business252(ql.NullCalendar())
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
            ql.PlainVanillaPayoff(ql.Option.Call, float(strike)),
            ql.EuropeanExercise(evaluation + days),
        )
        option.setPricingEngine(engine)
        return float(option.NPV())
    finally:
        ql.Settings.instance().evaluationDate = previous_evaluation


quantlib_heston_call_price = quantlib_call_price  # historical import compatibility


def call_price(spot, variance, maturity, strike, config):
    """Black--Scholes for GBM; Heston/Bates characteristic-function pricing."""
    if config.model == "gbm":
        from .gbm import gbm_call_price
        return gbm_call_price(spot, variance, maturity, strike, config)
    return heston_call_price(spot, variance, maturity, strike, config)


def transition(spot, variance, shocks=None, config=None, *, generator=None):
    """Shared conditional market interface; trading/accounting are separate."""
    config = config or HestonConfig()
    if config.model == "gbm":
        from .gbm import gbm_transition
        return gbm_transition(spot, variance, shocks, config, generator=generator)
    if config.model == "bates":
        from .bates import bates_transition
        return bates_transition(spot, variance, shocks, config, generator=generator)
    return heston_transition(spot, variance, shocks, config, generator=generator)


def market_shocks(config, rng, count=None):
    """Seeded NumPy chance draws for scalar/batched search, never future paths.

    Heston's three-channel draw order is unchanged. Bates appends an actual
    Poisson jump count and a normal aggregate-size shock; search cannot drop
    jumps by silently invoking the Heston transition.
    """
    columns = [rng.normal(size=count), rng.random(size=count), rng.normal(size=count)]
    if config.model == "bates":
        columns.extend((rng.poisson(config.jump_intensity * config.dt, size=count),
                        rng.normal(size=count)))
    return np.stack(columns, axis=-1)


def heston_transition(spot, variance, shocks=None, config: HestonConfig | None = None, *, generator=None):
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
    decay = math.exp(-config.kappa * config.dt)
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
    common = 0.5 * config.dt * (config.kappa * config.rho / config.sigma - 0.5)
    k0 = -config.rho * config.kappa * config.theta * config.dt / config.sigma
    k1, k2 = common - config.rho / config.sigma, common + config.rho / config.sigma
    noise_variance = 0.5 * config.dt * (1 - config.rho**2) * (variance + next_variance)
    next_spot = spot * torch.exp(k0 + k1 * variance + k2 * next_variance
                                + noise_variance.clamp_min(0).sqrt() * zs)
    return next_spot, next_variance


def mark_state(spot, variance, time_index, config: HestonConfig):
    """Current stock/call marks and positive liability value; no future inputs."""
    reference = spot.to(torch.float64)
    time = _tensor(time_index, reference) * config.dt
    hedge_time = _tensor(config.hedge_maturities, reference) - time[..., None]
    hedge_calls = call_price(reference[..., None], variance[..., None], hedge_time,
                                   config.hedge_strikes, config)
    liability_time = (config.n_steps * config.dt - time).clamp_min(0)
    liability = call_price(reference, variance, liability_time, config.liability_strike, config)
    return torch.cat((spot[..., None], hedge_calls.to(spot.dtype)), dim=-1), liability.to(spot.dtype)


marks_at = mark_state


@torch.no_grad()
def simulate_market_paths(config, n_paths: int, seed: int, *, device="cpu",
                          dtype=torch.float64, substeps=1):
    """Simulate at finer internal steps, returning only unchanged trading dates.

    The original config and observation/calendar contract do not change.
    substeps=1 preserves the historical Heston RNG order exactly. Different
    resolutions are not implicitly Brownian-coupled by sharing a seed.
    """
    if n_paths < 1 or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("positive path count and integer simulation substeps required")
    generator = torch.Generator(device=device).manual_seed(seed)
    spot = torch.full((n_paths,), config.spot0, device=device, dtype=torch.float64)
    variance = torch.full_like(spot, config.v0)
    internal = replace(config, dt=config.dt / substeps) if substeps != 1 else config
    spots, variances = [spot], [variance]
    for _ in range(config.n_steps):
        for _ in range(substeps):
            spot, variance = transition(spot, variance, config=internal, generator=generator)
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
    spot, variance = simulate_market_paths(config, n_paths, seed, device=device, substeps=substeps)
    return market_bank_from_paths(config, spot, variance, price_chunk_size=price_chunk_size, dtype=dtype)


def initial_ledger(reference: torch.Tensor, config: HestonConfig):
    """Zero initial holdings and the identical authoritative liability premium."""
    premium = call_price(_tensor(config.spot0, reference), config.v0,
                                config.n_steps * config.dt, config.liability_strike, config)
    zero = torch.zeros_like(reference)
    return LedgerState(zero + premium, zero[..., None].expand(*zero.shape, config.n_assets).clone(),
                       zero.clone(), zero[..., None].expand(*zero.shape, config.n_assets).clone(), zero.clone())


def initial_state(bank: MarketBank):
    zero = torch.zeros_like(bank.spot[:, 0])
    return LedgerState(bank.liability[:, 0].clone(), bank.marks[:, 0].new_zeros((len(zero), bank.config.n_assets)),
                       zero.clone(), bank.marks[:, 0].new_zeros((len(zero), bank.config.n_assets)), zero.clone())


def map_action(raw, config: HestonConfig):
    lower, upper = _tensor(config.holding_lower, raw), _tensor(config.holding_upper, raw)
    return lower + (upper - lower) * (torch.tanh(raw) + 1) / 2


def hybrid_action(modes, sizes, current_positions, config: HestonConfig):
    """Hold=0, buy=1, sell=2; sizes are legal fractions in [0,1]."""
    lower, upper = _tensor(config.holding_lower, sizes), _tensor(config.holding_upper, sizes)
    if bool(((sizes < 0) | (sizes > 1)).any()) or bool(((modes < 0) | (modes > 2)).any()):
        raise ValueError("hybrid action requires legal modes and unit interval sizes")
    return torch.where(modes == 1, current_positions + sizes * (upper - current_positions),
                       torch.where(modes == 2, current_positions - sizes * (current_positions - lower),
                                   current_positions))


def transaction_cost(trade, marks, config: HestonConfig):
    """Commission floor on nonzero trades, plus quadratic cost and fixed ticket.

    Minimum commission replaces a smaller proportional fee; it is not an
    additional fee. HOLD costs zero, and mandatory liquidation pays all fees.
    """
    commission = trade.abs() * marks * _tensor(config.proportional, marks)
    if any(config.minimum_commission):
        commission = torch.where(trade != 0,
                                 torch.maximum(commission, _tensor(config.minimum_commission, marks)),
                                 torch.zeros_like(commission))
    return (commission
            + trade.square() * marks * _tensor(config.quadratic, marks)
            + (trade != 0).to(marks.dtype) * _tensor(config.fixed_ticket, marks)).sum(-1)


def feasible_targets(previous, targets, config: HedgeBookConfig, *, liquidating=False):
    """Per-instrument legality, broadcasting over paths and candidate actions.

    Limits apply to trade increments, not desired target magnitudes. HOLD is
    always allowed within holding bounds. Mandatory terminal closeout waives
    the minimum-order size only; contract lots and all trading fees still apply.
    The tolerance covers floating-point representation, not economic slack.
    """
    previous, targets = torch.broadcast_tensors(previous, targets)
    lower, upper = _tensor(config.holding_lower, targets), _tensor(config.holding_upper, targets)
    valid = torch.isfinite(targets) & (targets >= lower) & (targets <= upper)
    if not any(config.minimum_trade + config.trade_lot):
        return valid
    trade = targets - previous
    hold = trade == 0
    lot = _tensor(config.trade_lot, targets)
    scale = torch.maximum(torch.maximum(targets.abs(), previous.abs()), lot)
    tolerance = 8 * torch.finfo(targets.dtype).eps * scale
    if not liquidating:
        valid = valid & (hold | (trade.abs() + tolerance >= _tensor(config.minimum_trade, targets)))
    if any(config.trade_lot):
        units = (trade / torch.where(lot > 0, lot, torch.ones_like(lot))).round()
        valid = valid & ((lot == 0) | hold | ((units != 0) & ((trade - units * lot).abs() <= tolerance)))
    return valid


def trade_step(state: LedgerState, target_positions, marks, config: HestonConfig, *, liquidating=False):
    """Execute legal targets without projection; retain legacy checkpoint behavior."""
    if config.execution_features and not bool(feasible_targets(
            state.positions, target_positions, config, liquidating=liquidating).all()):
        raise ValueError("target holdings violate holding bounds, minimum-trade and lot rules")
    trade = target_positions - state.positions
    costs = transaction_cost(trade, marks, config)
    return LedgerState(
        state.cash - (trade * marks).sum(-1) - costs, target_positions,
        state.total_cost + costs, state.turnover + trade.abs(),
        state.tickets + (trade != 0).sum(-1),
    )


def liquidate(state: LedgerState, terminal_marks, terminal_payoff, config: HestonConfig):
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


def numpy_ledger(marks, positions, initial_cash, liability_payoff, config: HestonConfig):
    """Independent sequential cash reconstruction, including the final sell-out.

    Deliberately uses scalar instrument loops and no Torch ledger helper.
    """
    marks, positions = np.asarray(marks), np.asarray(positions)
    cash = np.broadcast_to(np.asarray(initial_cash), (len(marks),)).astype(np.float64).copy()
    previous = np.zeros((len(marks), config.n_assets))
    costs, tickets = np.zeros(len(marks)), np.zeros(len(marks))
    turnover = np.zeros((len(marks), config.n_assets))
    for t in range(marks.shape[1]):
        target = positions[:, t] if t < positions.shape[1] else np.zeros_like(previous)
        for j in range(config.n_assets):
            trade = target[:, j] - previous[:, j]
            commission = config.proportional[j] * np.abs(trade) * marks[:, t, j]
            if config.minimum_commission[j]:
                commission = np.where(trade != 0, np.maximum(commission, config.minimum_commission[j]), 0)
            fee = (commission
                   + config.quadratic[j] * trade**2 * marks[:, t, j]
                   + config.fixed_ticket[j] * (trade != 0))
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


def observation_fields(config: HedgeBookConfig):
    instruments = ("stock", *(f"call_{i+1}" for i in range(config.n_assets - 1)))
    return (
        "time_fraction", "spot", "variance", "cash",
        *(f"{name}_position" for name in instruments),
        *(f"{name}_mid" for name in instruments),
        *(f"{kind}_{j}" for kind in ("proportional", "quadratic", "fixed_ticket",
                                    "holding_lower", "holding_upper") for j in range(config.n_assets)),
        *(f"{kind}_{j}" for kind in (("minimum_commission", "minimum_trade", "trade_lot")
                                    if config.execution_features else ()) for j in range(config.n_assets)),
        *market_parameter_names(config),
        "liability_strike", "liability_maturity",
        *(f"hedge_strike_{i+1}" for i in range(config.n_assets - 1)),
        *(f"hedge_maturity_{i+1}" for i in range(config.n_assets - 1)), "dt", "spot0", "v0",
    )


OBSERVATION_FIELDS = observation_fields(HestonConfig())


def observation_from_state(spot, variance, time_index, state: LedgerState, marks, config: HestonConfig):
    """Only current information. Cash preserves wealth for static terminal ES.

    Values use S0 and v0 scales where appropriate. Append the episode-global
    ES threshold outside this financial state when the risk method needs it.
    """
    time = torch.zeros_like(spot) + _tensor(time_index, spot) / config.n_steps
    values = [time[..., None], (spot / config.spot0)[..., None],
              (variance / max(config.v0, 1e-4))[..., None], (state.cash / config.spot0)[..., None],
              state.positions, marks / config.spot0]
    for name in ("proportional", "quadratic", "fixed_ticket", "holding_lower", "holding_upper"):
        values.append(_tensor(getattr(config, name), spot).expand(*spot.shape, config.n_assets))
    if config.execution_features:
        for name in ("minimum_commission", "minimum_trade", "trade_lot"):
            values.append(_tensor(getattr(config, name), spot).expand(*spot.shape, config.n_assets))
    market_parameters = tuple(getattr(config, name) for name in market_parameter_names(config))
    constants = (*market_parameters,
                 config.liability_strike, config.n_steps * config.dt, *config.hedge_strikes,
                 *config.hedge_maturities, config.dt, config.spot0, config.v0)
    values.append(_tensor(constants, spot).expand(*spot.shape, len(constants)))
    return torch.cat(values, dim=-1)


def observation(bank: MarketBank, time_index: int, state: LedgerState):
    return observation_from_state(bank.spot[:, time_index], bank.variance[:, time_index],
                                  time_index, state, bank.marks[:, time_index], bank.config)
