"""Configuration independence and regressions at the financial boundary."""
from dataclasses import asdict, replace

import pytest
import torch

from hedging_gym import (
    BatesConfig, EuropeanOption, ExecutionConfig, GBMConfig, HestonConfig,
    PortfolioConfig, TimeGrid, benchmark_config, config_from_dict, operational_config,
    RiskConfig,
)
from hedging_gym import finance


def test_mse_objective_roundtrip_value_and_gradient():
    config = benchmark_config(risk=RiskConfig(objective="mse"))
    restored = config_from_dict(asdict(config))
    assert restored == config
    losses = torch.tensor([-2., 0., 3.], requires_grad=True)
    value = restored.risk.loss(losses).mean()
    torch.testing.assert_close(value, torch.tensor(13./3))
    value.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([-4./3, 0., 2.]))
    torch.testing.assert_close(restored.risk.reward(losses), -losses.square())


def test_model_identity_and_market_only_simulation():
    with pytest.raises(TypeError):
        HestonConfig(model="bates")
    for market in (GBMConfig(), HestonConfig(), BatesConfig()):
        config = benchmark_config(model=market)
        assert config_from_dict(asdict(config)) == config
    # A year of market paths has no dependency on a 60-day hedge contract.
    spot, variance = finance.simulate_market_paths(GBMConfig(), TimeGrid(n_steps=252), 2, 11)
    assert spot.shape == variance.shape == (2, 253)
    assert torch.isfinite(spot).all() and (spot > 0).all()


def test_source_market_defaults_and_explicit_saved_parameters():
    heston = benchmark_config().market
    assert (heston.v0, heston.kappa, heston.theta, heston.sigma, heston.rho) == (.04, 1., .04, 2., -.7)
    assert benchmark_config(model="gbm").market == GBMConfig(v0=.09, mu=0.)
    # Saved banks/checkpoints carry explicit coefficients, not today's defaults.
    legacy = benchmark_config(model=HestonConfig(kappa=3., sigma=.3, rho=-.5))
    assert config_from_dict(asdict(legacy)) == legacy
    assert config_from_dict(asdict(legacy)).market != heston


def test_execution_broadcast_and_schema_do_not_depend_on_construction_order():
    config = benchmark_config(execution=ExecutionConfig(proportional=.01))
    extra_option = EuropeanOption(.95, 90/252, "put")
    larger_book = replace(config.portfolio, hedges=(*config.portfolio.hedges, extra_option))
    expanded = replace(config, portfolio=larger_book)
    assert expanded.execution.vector("proportional", expanded.n_assets) == (.01, .01, .01)
    assert expanded.execution.vector("minimum_trade", expanded.n_assets) == (0., 0., 0.)
    overlaid = operational_config(expanded, "operational_minimum_fee")
    assert finance.observation_fields(expanded) == finance.observation_fields(overlaid)
    assert overlaid.portfolio == larger_book and overlaid.market == config.market
    # User-specified per-instrument charges must not be silently resized.
    with pytest.raises(ValueError, match="proportional needs one value"):
        replace(expanded, execution=ExecutionConfig(proportional=(.01, .02)))


def test_stock_only_signed_put_liability_and_unconditional_limits():
    grid = TimeGrid(n_steps=2)
    portfolio = PortfolioConfig(EuropeanOption(1., grid.horizon, "put"), liability_quantity=-2.)
    config = benchmark_config(model="gbm", time_grid=grid, portfolio=portfolio,
                              execution=ExecutionConfig(holding_lower=0., holding_upper=0.))
    bank = finance.generate_market_bank(config, 3, 7, dtype=torch.float64)
    state = finance.initial_state(bank)
    assert state.positions.shape == (3, 1) and bank.marks.shape == (3, 3, 1)
    torch.testing.assert_close(bank.liability[:, -1], -2 * (1. - bank.spot[:, -1]).clamp_min(0))
    with pytest.raises(ValueError, match="holding bounds"):
        finance.trade_step(state, torch.ones_like(state.positions), bank.marks[:, 0], config)
    with pytest.raises(ValueError, match="holding bounds"):
        finance.ledger_from_positions(bank, torch.ones((3, 2, 1), dtype=torch.float64))
    result = finance.ledger_from_positions(bank, torch.zeros((3, 2, 1), dtype=torch.float64))
    torch.testing.assert_close(result["terminal_loss"], bank.liability[:, -1] - bank.liability[:, 0])


@pytest.mark.parametrize("market", [GBMConfig(), HestonConfig(), BatesConfig()])
def test_put_and_call_prices_match_quantlib_on_supported_clocks(market):
    # Test the new payoff/calendar composition, not a new pricing algorithm.
    for days_per_year in (252, 365, 360):
        grid = TimeGrid(n_steps=15, days_per_year=days_per_year)
        for kind in ("call", "put"):
            value = finance.option_price(torch.tensor(1., dtype=torch.float64), .04,
                                         grid.horizon, 1.03, market, kind=kind)
            reference = finance.quantlib_option_price(1., .04, grid.horizon, 1.03, market,
                                                   kind=kind, days_per_year=days_per_year)
            assert float(value) == pytest.approx(reference, abs=1e-8, rel=1e-7)


def test_hedge_settlement_at_horizon_and_named_market_state():
    grid = TimeGrid(n_steps=2, days_per_year=365)
    liability = EuropeanOption(100., grid.horizon)
    config = benchmark_config(model=GBMConfig(spot0=100.), time_grid=grid,
        portfolio=PortfolioConfig(liability, (EuropeanOption(105., grid.horizon, "put"),)))
    bank = finance.generate_market_bank(config, 2, 17, dtype=torch.float64)
    torch.testing.assert_close(bank.marks[:, -1, 1], (105. - bank.spot[:, -1]).clamp_min(0))
    observed = finance.observation(bank, 0, finance.initial_state(bank))
    spot, variance = finance.decode_market_observation(observed, config)
    torch.testing.assert_close(spot, bank.spot[:, 0])
    torch.testing.assert_close(variance, bank.variance[:, 0])


def test_expiry_is_exact_for_equivalent_year_fraction_expressions():
    # Division and multiplication differ by a few ulps for these dates.
    # That must not change contract eligibility or request near-zero-time pricing.
    for days, basis in ((3, 365), (33, 252)):
        option = EuropeanOption(1., days / basis)
        config = benchmark_config(time_grid=TimeGrid(days, basis),
                                  portfolio=PortfolioConfig(option, (option,)))
        bank = finance.generate_market_bank(config, 1, 7, dtype=torch.float64)
        payoff = (bank.spot[:, -1] - option.strike).clamp_min(0)
        torch.testing.assert_close(bank.liability[:, -1], payoff, rtol=0, atol=0)
        torch.testing.assert_close(bank.marks[:, -1, 1], payoff, rtol=0, atol=0)
