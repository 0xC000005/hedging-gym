"""Independent market, portfolio, execution and risk settings.

Benchmark defaults belong in benchmark.py. These small value objects describe
the supported financial contract without fixing the number of hedge options.
"""
from dataclasses import dataclass, field
import math
from numbers import Real


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
    kappa: float = 3.0
    theta: float = 0.04
    sigma: float = 0.3
    rho: float = -0.5
    model: str = field(default="heston", init=False)

    def __post_init__(self):
        super().__post_init__()
        if (not all(math.isfinite(x) for x in (self.kappa, self.theta, self.sigma, self.rho))
                or min(self.kappa, self.theta, self.sigma) <= 0 or not -1 <= self.rho <= 1):
            raise ValueError("invalid Heston coefficients")


@dataclass(frozen=True)
class GBMConfig(MarketConfig):
    """mu is physical stock drift; it does not enter risk-neutral pricing."""
    mu: float = 0.0
    model: str = field(default="gbm", init=False)

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.mu) or self.v0 <= 0:
            raise ValueError("GBM requires finite drift and positive variance")


@dataclass(frozen=True)
class BatesConfig(HestonConfig):
    """Heston plus independent compensated normal log jumps."""
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

    def __post_init__(self):
        if not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("positive integer trading-step count required")
        if self.days_per_year not in (252, 365, 360):
            raise ValueError("supported year clocks are 252, 365 and 360 days")

    @property
    def dt(self):
        return 1.0 / self.days_per_year

    @property
    def horizon(self):
        return self.n_steps * self.dt


@dataclass(frozen=True)
class EuropeanOption:
    strike: float
    maturity: float  # years on the selected TimeGrid clock
    kind: str = "call"

    def __post_init__(self):
        if (self.kind not in ("call", "put") or not math.isfinite(self.strike)
                or not math.isfinite(self.maturity) or min(self.strike, self.maturity) <= 0):
            raise ValueError("European call/put requires positive finite strike and maturity")


@dataclass(frozen=True)
class PortfolioConfig:
    """One underlying, any number of hedge calls/puts, and a signed liability.

    A positive liability_quantity is owed; a negative quantity is owned. Hedge
    positions are numbers of options, each with unit underlying multiplier.
    Cash is accounted separately and is not an action-space instrument.
    """
    liability: EuropeanOption
    hedges: tuple[EuropeanOption, ...] = ()
    liability_quantity: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "hedges", tuple(self.hedges))
        if not math.isfinite(self.liability_quantity):
            raise ValueError("liability quantity must be finite")

    @property
    def n_assets(self):
        return 1 + len(self.hedges)


EXECUTION_FIELDS = ("proportional", "quadratic", "fixed_ticket", "minimum_commission",
                    "minimum_trade", "trade_lot", "holding_lower", "holding_upper")


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
    holding_lower: float | tuple[float, ...] = -1.0
    holding_upper: float | tuple[float, ...] = 1.0

    def __post_init__(self):
        for name in EXECUTION_FIELDS:
            value = getattr(self, name)
            values = (value,) if isinstance(value, Real) else tuple(value)
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"{name} must be finite")
            if name not in ("holding_lower", "holding_upper") and any(x < 0 for x in values):
                raise ValueError(f"{name} must be nonnegative")
            if not isinstance(value, Real):
                object.__setattr__(self, name, values)

    def vector(self, name, n_assets):
        value = getattr(self, name)
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
    """Terminal expected-shortfall confidence; zeta is learned separately."""
    alpha: float = 0.95

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("risk confidence must be between zero and one")

    def loss(self, losses, zeta):
        """Rockafellar--Uryasev loss, averaged across complete episodes by caller."""
        return zeta + (losses - zeta).relu() / (1 - self.alpha)


@dataclass(frozen=True)
class HedgingConfig:
    market: GBMConfig | HestonConfig | BatesConfig
    time_grid: TimeGrid
    portfolio: PortfolioConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

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
        if any(round(option.maturity * self.time_grid.days_per_year) < self.n_steps
               for option in self.portfolio.hedges):
            raise ValueError("hedges cannot expire before the episode; intermediate settlement is unsupported")
        self.execution.validate(self.n_assets)

    @property
    def n_assets(self):
        return self.portfolio.n_assets

    @property
    def n_steps(self):
        return self.time_grid.n_steps

    @property
    def dt(self):
        return self.time_grid.dt


def config_from_dict(values):
    """Load the explicit nested configuration emitted by dataclasses.asdict."""
    book = values["portfolio"]
    return HedgingConfig(
        market=market_from_dict(values["market"]), time_grid=TimeGrid(**values["time_grid"]),
        portfolio=PortfolioConfig(EuropeanOption(**book["liability"]),
                                  tuple(EuropeanOption(**option) for option in book["hedges"]),
                                  book["liability_quantity"]),
        execution=ExecutionConfig(**values["execution"]), risk=RiskConfig(**values["risk"]),
    )
