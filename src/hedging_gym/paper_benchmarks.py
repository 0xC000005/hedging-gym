"""Published financial tasks composed from ordinary environment settings.

These preserve the stated economic problems, not author RNG streams, neural
architectures or training results. See docs/paper-benchmarks.md for provenance
and the numerical/reconstruction differences.
"""
from .config import (EuropeanOption, ExecutionConfig, GBMConfig, HedgingConfig,
                     HestonConfig, PortfolioConfig, RiskConfig, SettlementConfig, TimeGrid)
from .instruments import VarianceSwap


def buehler_heston(*, objective="es", alpha=.5):
    """Deep Hedging §§5.1–5.2: stock + variance swap, 30/365, no fees.

    MSE selects the single-Heston variant in §5.4. Unlimited holdings are
    intentional: the raw unannualized variance leg is very small in price units.
    """
    grid = TimeGrid(n_steps=30, days_per_year=365)
    return HedgingConfig(
        market=HestonConfig(spot0=100.), time_grid=grid,
        portfolio=PortfolioConfig(EuropeanOption(100., grid.horizon),
                                  (VarianceSwap(grid.horizon),)),
        execution=ExecutionConfig(holding_lower=None, holding_upper=None),
        risk=RiskConfig(objective=objective, alpha=alpha),
        settlement=SettlementConfig(charge_liquidation_costs=False),
    )


def szehr_market(model="heston", *, objective="mse"):
    """Released minimalHedger 2023 market/book/calendar, with raw MSE.

    Heston uses 60/365 and GBM 30/365, stock + cash, no hedge option. This
    deliberately does not copy the source's clipped reward or hardcoded premium.
    """
    if model not in ("heston", "gbm"):
        raise ValueError("select heston or gbm")
    grid = TimeGrid(n_steps=60 if model == "heston" else 30, days_per_year=365)
    return HedgingConfig(
        market=HestonConfig() if model == "heston" else GBMConfig(), time_grid=grid,
        portfolio=PortfolioConfig(EuropeanOption(1., grid.horizon)),
        risk=RiskConfig(objective=objective),
    )


def maggiolo_gbm(*, step_days):
    """AlphaZero vs Deep Hedging §4.2.2: five actions, four GBM moves.

    The paper omits physical dt: the caller MUST declare step_days on a 365-day
    clock. step_days=365 is the unit-time reconstruction, not a sourced fact.
    Twenty target holdings, including the maturity-date trade, are enforced by
    bounds and 0.05-share lots. Initial wealth is 0.4 cash + 0.4*5 stock = 2.4.
    """
    grid = TimeGrid(n_steps=4, days_per_year=365, step_days=step_days, trade_at_maturity=True)
    return HedgingConfig(
        market=GBMConfig(spot0=5., v0=.25**2, mu=.03125), time_grid=grid,
        portfolio=PortfolioConfig(EuropeanOption(5., grid.horizon),
                                  initial_cash=.4, initial_positions=(.4,)),
        execution=ExecutionConfig(holding_lower=0., holding_upper=.95, trade_lot=.05,
                                  per_unit=.25, commission_cap=.05),
        risk=RiskConfig(objective="mse"),
        settlement=SettlementConfig(mode="mark_to_market"),
    )
