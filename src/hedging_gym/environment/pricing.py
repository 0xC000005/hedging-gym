"""Scalar QuantLib marking for adapters that query one market state at a time.

The batched backend remains ``finance.mark_state``. Both use the configured
contracts and calendar; the scalar backend skips unused interim liability marks.
"""

from . import finance
from .instruments import EuropeanOption


def quantlib_mark_state(spot, variance, date, config, *, integrated_variance=None):
    """Return hedge marks and, at settlement, the signed liability value.

    QuantLib's Heston model requires positive initial variance. At zero variance
    use the tensor pricing formula rather than alter the supplied market state.
    """
    def price(option):
        remaining = (round(option.maturity * config.time_grid.days_per_year)
                     - min(date, config.n_steps) * config.time_grid.step_days) / config.time_grid.days_per_year
        if not isinstance(option, EuropeanOption):
            return float(option.mark(spot, variance, remaining, config.market,
                                     integrated_variance=integrated_variance)[0])
        if float(variance[0]) == 0.:
            return float(finance.option_price(spot, variance, remaining,
                         option.strike, config.market, kind=option.kind)[0])
        return finance.quantlib_option_price(float(spot[0]), float(variance[0]),
            remaining, option.strike, config.market,
            days_per_year=config.time_grid.days_per_year, kind=option.kind)

    marks = spot.new_tensor([[float(spot[0]), *(price(option) for option in config.portfolio.hedges)]])
    liability = (spot.new_tensor([config.portfolio.liability_quantity * price(config.portfolio.liability)])
                 if date >= config.n_steps else None)
    return marks, liability
