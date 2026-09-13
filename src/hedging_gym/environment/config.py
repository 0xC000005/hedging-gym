"""Independent market, portfolio, execution and risk settings.

Market defaults follow the sources documented in docs/benchmark.md. Portfolio
and execution defaults belong in benchmark.py, independently of these models.
"""
from dataclasses import dataclass, field
import math
from numbers import Real
from .instruments import EuropeanOption, Instrument, VarianceSwap, instrument_from_dict


@dataclass(frozen=True)
class MarketConfig:
    spot0: float = 1.0
    v0: float = 0.04
    r: float = 0.0
    q: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.spot0, self.v0, self.r, self.q)):
            raise ValueError("market parameters must be finite")
        if self.spot0 <= 0 or self.v0 < 0:
            raise ValueError("positive spot and nonnegative initial variance required")
        if self.r != 0 or self.q != 0:
            raise ValueError("financing and dividends are not implemented; r=q=0 required")


@dataclass(frozen=True)
class HestonConfig(MarketConfig):
    """Deep Hedging §5.2 / minimalHedger Heston parameters, with normalized spot."""
    kappa: float = 1.0
    theta: float = 0.04
    sigma: float = 2.0
    rho: float = -0.7
    scheme: str = "qe_m"
    model: str = field(default="heston", init=False)

    def __post_init__(self):
        super().__post_init__()
        if (not all(math.isfinite(x) for x in (self.kappa, self.theta, self.sigma, self.rho))
                or min(self.kappa, self.theta, self.sigma) <= 0 or not -1 <= self.rho <= 1):
            raise ValueError("invalid Heston coefficients")
        if self.scheme not in ("qe", "qe_m"):
            raise ValueError("Heston scheme must be qe or qe_m")


@dataclass(frozen=True)
class GBMConfig(MarketConfig):
    """minimalHedger's 30% volatility; mu is physical, not pricing, drift."""
    v0: float = 0.09  # Variance, i.e. 0.3**2, not volatility.
    mu: float = 0.0
    model: str = field(default="gbm", init=False)

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.mu) or self.v0 <= 0:
            raise ValueError("GBM requires finite drift and positive variance")


@dataclass(frozen=True)
class BatesConfig(HestonConfig):
    """Heston plus independent compensated normal log jumps."""
    # Keep the existing Bates scenario independent of Heston benchmark defaults.
    kappa: float = 3.0
    sigma: float = 0.3
    rho: float = -0.5
    jump_intensity: float = 1.0
    jump_mean: float = -0.1
    jump_std: float = 0.2
    model: str = field(default="bates", init=False)

    def __post_init__(self):
        super().__post_init__()
        if (not all(math.isfinite(x) for x in (self.jump_intensity, self.jump_mean, self.jump_std))
                or min(self.jump_intensity, self.jump_std) < 0):
            raise ValueError("invalid Bates jump coefficients")


def market_from_dict(values):
    """Construct the selected market, using current defaults for omitted fields."""
    parameters = dict(values)
    model = parameters.pop("model")
    classes = {"gbm": GBMConfig, "heston": HestonConfig, "bates": BatesConfig}
    if model not in classes:
        raise ValueError(f"unknown market model: {model}")
    return classes[model](**parameters)


@dataclass(frozen=True)
class TimeGrid:
    """Equally spaced trading dates, with an explicit reference year clock."""
    n_steps: int = 30
    days_per_year: int = 252
    step_days: int = 1
    trade_at_maturity: bool = False

    def __post_init__(self):
        if not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("positive integer trading-step count required")
        if self.days_per_year not in (252, 365, 360):
            raise ValueError("supported year clocks are 252, 365 and 360 days")
        if not isinstance(self.step_days, int) or self.step_days < 1:
            raise ValueError("positive integer days between market steps required")

    @property
    def dt(self):
        return self.step_days / self.days_per_year

    @property
    def horizon(self):
        return self.n_steps * self.dt


@dataclass(frozen=True)
class PortfolioConfig:
    """One underlying, user-supplied priced hedge instruments and a liability.

    A positive liability_quantity is owed; a negative quantity is owned. Hedge
    positions are numbers of options, each with unit underlying multiplier.
    Cash is accounted separately and is not an action-space instrument.
    """
    liability: Instrument
    hedges: tuple[Instrument, ...] = ()
    liability_quantity: float = 1.0
    initial_cash: float | None = None  # None uses the model liability premium.
    initial_positions: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "hedges", tuple(self.hedges))
        if not math.isfinite(self.liability_quantity):
            raise ValueError("liability quantity must be finite")
        if self.initial_cash is not None and not math.isfinite(self.initial_cash):
            raise ValueError("initial cash must be finite")
        if self.initial_positions is not None:
            object.__setattr__(self, "initial_positions", tuple(self.initial_positions))
            if len(self.initial_positions) != self.n_assets or not all(map(math.isfinite, self.initial_positions)):
                raise ValueError("initial positions need one finite quantity per instrument")

    @property
    def n_assets(self):
        return 1 + len(self.hedges)


EXECUTION_FIELDS = ("proportional", "quadratic", "fixed_ticket", "minimum_commission",
                    "minimum_trade", "trade_lot", "holding_lower", "holding_upper",
                    "per_unit", "commission_cap")


@dataclass(frozen=True)
class ExecutionConfig:
    """Scalars apply to every instrument; tuples follow the portfolio order.

    Scalars stay scalars when a portfolio changes. Explicit tuples must match
    the selected book; we never guess how to resize user-specified charges.
    """
    proportional: float | tuple[float, ...] = 0.0
    quadratic: float | tuple[float, ...] = 0.0
    fixed_ticket: float | tuple[float, ...] = 0.0
    minimum_commission: float | tuple[float, ...] = 0.0
    minimum_trade: float | tuple[float, ...] = 0.0
    trade_lot: float | tuple[float, ...] = 0.0
    holding_lower: float | tuple[float, ...] | None = -1.0
    holding_upper: float | tuple[float, ...] | None = 1.0
    per_unit: float | tuple[float, ...] = 0.0
    commission_cap: float | tuple[float, ...] = 0.0  # Zero means uncapped.

    def __post_init__(self):
        for name in EXECUTION_FIELDS:
            value = getattr(self, name)
            if value is None and name in ("holding_lower", "holding_upper"):
                continue
            values = (value,) if isinstance(value, Real) else tuple(value)
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"{name} must be finite")
            if name not in ("holding_lower", "holding_upper") and any(x < 0 for x in values):
                raise ValueError(f"{name} must be nonnegative")
            if not isinstance(value, Real):
                object.__setattr__(self, name, values)

    def vector(self, name, n_assets):
        value = getattr(self, name)
        if value is None and name in ("holding_lower", "holding_upper"):
            return ((-math.inf if name == "holding_lower" else math.inf),) * n_assets
        values = (float(value),) * n_assets if isinstance(value, Real) else value
        if len(values) != n_assets:
            raise ValueError(f"{name} needs one value per instrument ({n_assets})")
        return values

    def validate(self, n_assets):
        vectors = {name: self.vector(name, n_assets) for name in EXECUTION_FIELDS}
        if any(lo > 0 or hi < 0 or lo > hi for lo, hi in
               zip(vectors["holding_lower"], vectors["holding_upper"])):
            raise ValueError("holding bounds must contain the zero initial portfolio")


@dataclass(frozen=True)
class RiskConfig:
    """Terminal objective; ES and entropy need a global threshold learned separately.

    The entropic risk log E[exp(risk_aversion * loss)] / risk_aversion is the
    exponential-utility certainty equivalent (Föllmer and Schied). Its optimized
    certainty equivalent form (Ben-Tal and Teboulle 2007) shares the ES
    threshold mechanics: minimizing over zeta returns the risk itself.
    """
    alpha: float = 0.95
    objective: str = "es"
    risk_aversion: float = 1.

    def __post_init__(self):
        if self.objective not in ("es", "mse", "entropy"):
            raise ValueError("choose terminal es, mse or entropy")
        if not 0 < self.alpha < 1:
            raise ValueError("risk confidence must be between zero and one")
        if self.risk_aversion <= 0:
            raise ValueError("entropic risk aversion must be positive")

    def loss(self, losses, zeta=None):
        """Per-path loss; the learner averages over complete episodes."""
        if self.objective == "mse":
            return losses.square()
        if zeta is None:
            raise ValueError("ES and entropic losses require a global risk threshold")
        if self.objective == "entropy":
            return zeta + (self.risk_aversion * (losses - zeta)).expm1() / self.risk_aversion
        return zeta + (losses - zeta).relu() / (1 - self.alpha)

    def entropic_risk(self, losses):
        """Pooled entropic risk of complete-episode losses, in loss units."""
        scaled = self.risk_aversion * losses
        return (scaled.logsumexp(0) - math.log(len(losses))) / self.risk_aversion

    def reward(self, losses, zeta=None):
        """Gym reward: negative selected loss, or raw P&L before threshold calibration."""
        if self.objective != "mse" and zeta is None:
            return -losses
        return -self.loss(losses, zeta)


@dataclass(frozen=True)
class SettlementConfig:
    """Close hedge positions with a trade, or value the remaining inventory."""
    mode: str = "liquidate"
    charge_liquidation_costs: bool = True

    def __post_init__(self):
        if self.mode not in ("liquidate", "mark_to_market"):
            raise ValueError("settlement mode must be liquidate or mark_to_market")


@dataclass(frozen=True)
class HedgingConfig:
    market: GBMConfig | HestonConfig | BatesConfig
    time_grid: TimeGrid
    portfolio: PortfolioConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    settlement: SettlementConfig = field(default_factory=SettlementConfig)

    def __post_init__(self):
        if not isinstance(self.market, (GBMConfig, HestonConfig)):
            raise ValueError("choose a GBM, Heston or Bates market")
        if not math.isclose(self.portfolio.liability.maturity, self.time_grid.horizon,
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("liability must settle at the episode horizon")
        # Reference engines use dates. Reject unsupported fractional-day contracts
        # here, rather than running a benchmark its independent oracle cannot price.
        for option in (self.portfolio.liability, *self.portfolio.hedges):
            days = option.maturity * self.time_grid.days_per_year
            if not math.isclose(days, round(days), rel_tol=0, abs_tol=1e-10):
                raise ValueError("contract maturities must lie on the selected year clock")
        if any(option.maturity < self.time_grid.horizon - 1e-12
               for option in self.portfolio.hedges):
            raise ValueError("hedges cannot expire before the episode; intermediate settlement is unsupported")
        self.execution.validate(self.n_assets)
        positions = self.portfolio.initial_positions or (0.,) * self.n_assets
        if any(not lo <= pos <= hi for lo, pos, hi in zip(
                self.execution.vector("holding_lower", self.n_assets), positions,
                self.execution.vector("holding_upper", self.n_assets))):
            raise ValueError("initial positions must satisfy holding bounds")

    @property
    def n_assets(self):
        return self.portfolio.n_assets

    @property
    def n_steps(self):
        return self.time_grid.n_steps

    @property
    def dt(self):
        return self.time_grid.dt

    @property
    def n_decisions(self):
        return self.n_steps + int(self.time_grid.trade_at_maturity)


def config_from_dict(values):
    """Load the explicit nested configuration emitted by dataclasses.asdict."""
    book = values["portfolio"]
    return HedgingConfig(
        market=market_from_dict(values["market"]), time_grid=TimeGrid(**values["time_grid"]),
        portfolio=PortfolioConfig(instrument_from_dict(book["liability"]),
                                  tuple(instrument_from_dict(option) for option in book["hedges"]),
                                  book["liability_quantity"], book.get("initial_cash"),
                                  book.get("initial_positions")),
        execution=ExecutionConfig(**values["execution"]), risk=RiskConfig(**values["risk"]),
        settlement=SettlementConfig(**values.get("settlement", {})),
    )
