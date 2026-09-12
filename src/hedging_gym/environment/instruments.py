"""Priced claims plugged into the common market and cash ledger.

An instrument supplies its current value (its payoff at expiry), numeric
contract features, and whether it needs accumulated variance. It does not
simulate markets, execute trades or change the risk objective. Methods must
treat the supplied tensors as read-only and preserve their batch dimensions.
"""
from dataclasses import dataclass, field
import math
from typing import Protocol


class Instrument(Protocol):
    maturity: float
    kind: str
    needs_integrated_variance: bool

    @property
    def features(self) -> dict[str, float]: ...

    def mark(self, spot, variance, remaining, market, *, integrated_variance=None): ...


@dataclass(frozen=True)
class EuropeanOption:
    strike: float
    maturity: float
    kind: str = "call"
    needs_integrated_variance = False

    def __post_init__(self):
        if (self.kind not in ("call", "put") or not math.isfinite(self.strike)
                or not math.isfinite(self.maturity) or min(self.strike, self.maturity) <= 0):
            raise ValueError("European call/put requires positive finite strike and maturity")

    @property
    def features(self):
        return dict(strike=self.strike, maturity=self.maturity, kind=1. if self.kind == "call" else -1.)

    def mark(self, spot, variance, remaining, market, *, integrated_variance=None):
        from .finance import option_price
        return option_price(spot, variance, remaining, self.strike, market, kind=self.kind)


@dataclass(frozen=True)
class VarianceSwap:
    """Bühler's unannualized variance leg, paying integral_0^T V(t) dt.

    No annualization or strike subtraction: this is the positive-price claim
    used in the paper, not the usual zero-value struck swap quotation.
    """
    maturity: float
    kind: str = field(default="variance_swap", init=False)
    needs_integrated_variance = True

    def __post_init__(self):
        if not math.isfinite(self.maturity) or self.maturity <= 0:
            raise ValueError("positive finite variance-swap maturity required")

    @property
    def features(self):
        return dict(maturity=self.maturity)

    def mark(self, spot, variance, remaining, market, *, integrated_variance=None):
        from .finance import variance_swap_price
        if integrated_variance is None:
            raise ValueError("variance-swap marks require accumulated realized variance")
        return variance_swap_price(variance, integrated_variance, remaining, market).to(spot.dtype)


_INSTRUMENT_FACTORIES = {
    "call": lambda **values: EuropeanOption(kind="call", **values),
    "put": lambda **values: EuropeanOption(kind="put", **values),
    "variance_swap": VarianceSwap,
}


def register_instrument(kind, factory):
    """Register a custom dataclass constructor for config/checkpoint loading.

    Passing a custom Instrument directly needs no registration. Register its
    unique kind in the loading process only when using serialized configs.
    """
    if kind in _INSTRUMENT_FACTORIES:
        raise ValueError(f"instrument kind already registered: {kind}")
    _INSTRUMENT_FACTORIES[kind] = factory


def instrument_from_dict(values):
    parameters = dict(values)
    kind = parameters.pop("kind", "call")
    return _INSTRUMENT_FACTORIES[kind](**parameters)
