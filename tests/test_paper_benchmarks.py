"""Published task composition, extensible instruments and terminal accounting."""
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pytest
import torch

from hedging_gym import (
    ExecutionConfig,
    GBMConfig,
    HedgingConfig,
    PortfolioConfig,
    RiskConfig,
    SettlementConfig,
    TensorHedgingEnv,
    TimeGrid,
    config_from_dict,
    register_instrument,
)
from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.baselines.delta_variance import (
    delta_variance_hedge_positions,
    spot_variance_greeks,
)
from hedging_gym.baselines.source_alphazero import SourceHedgingGame
from hedging_gym.environment import finance
from hedging_gym.environment.paper_benchmarks import (
    buehler_heston,
    maggiolo_gbm,
    szehr_market,
)


def test_buehler_variance_swap_greeks_and_shared_accounting():
    config = buehler_heston(objective="mse")
    assert config_from_dict(asdict(config)) == config
    bank = finance.generate_market_bank(config, 4, 91, dtype=torch.float64, substeps=2)
    # Known conditional value at inception, actual accumulated payoff at expiry.
    torch.testing.assert_close(bank.marks[:, 0, 1], torch.full((4,), .04*30/365, dtype=torch.float64))
    torch.testing.assert_close(bank.marks[:, -1, 1], bank.integrated_variance[:, -1], rtol=0, atol=0)
    reference = finance.quantlib_option_price(100., .04, 30/365, 100., config.market, days_per_year=365)
    assert float(bank.liability[0, 0]) == pytest.approx(reference, abs=1e-6)
    ds, dv = spot_variance_greeks(bank.spot[:, :1], bank.variance[:, :1], 0, config,
                                 integrated_variance=bank.integrated_variance[:, :1])
    torch.testing.assert_close(ds[..., 1], torch.zeros_like(ds[..., 1]), rtol=0, atol=0)
    torch.testing.assert_close(dv[..., 1], torch.full_like(dv[..., 1], -np.expm1(-30/365)))
    targets = delta_variance_hedge_positions(bank)
    actual = finance.ledger_from_positions(bank, targets)
    expected = finance.numpy_ledger(bank.marks.numpy(), targets.numpy(),
        finance.initial_state(bank).cash.numpy(), bank.liability[:, -1].numpy(), config)
    np.testing.assert_allclose(actual["terminal_loss"], expected["terminal_loss"], atol=1e-12, rtol=1e-12)
    assert torch.isfinite(targets).all() and torch.count_nonzero(actual["total_cost"]) == 0
    observed = finance.observation(bank, 0, finance.initial_state(bank))
    assert torch.isfinite(observed).all()
    policy = DirectDHPolicy(config, hidden=(8,)).double()
    action = policy(observed, finance.initial_state(bank).positions, None, None)
    action.target_holdings.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in policy.parameters())


def test_maggiolo_terminal_action_cash_endowment_and_capped_fee():
    config = maggiolo_gbm(step_days=365)
    assert config_from_dict(asdict(config)) == config
    assert config.n_steps == 4 and config.n_decisions == 5
    spot = torch.tensor([[5., 5.5, 4.9, 6., 5.8], [5., 4.5, 5.2, 4., 4.7]], dtype=torch.float64)
    bank = finance.market_bank_from_paths(config, spot, torch.full_like(spot, .25**2))
    targets = torch.tensor([[[.5], [.55], [.3], [.8], [.6]],
                            [[.4], [.2], [.35], [.5], [.5]]], dtype=torch.float64)
    env = TensorHedgingEnv(bank)
    for t in range(5):
        _, reward, done, _, result = env.step(targets[:, t])
        assert done == (t == 4)
    previous = torch.cat((torch.full((2, 1), .4, dtype=torch.float64), targets[:, :-1, 0]), dim=1)
    cost = torch.minimum(.25*(targets[..., 0]-previous).abs(), torch.tensor(.05, dtype=torch.float64)).sum(-1)
    # Independent wealth-increment form from the paper's accounting equation.
    wealth = 2.4 + (targets[:, :4, 0] * spot.diff(dim=1)).sum(-1) - cost
    expected_loss = (spot[:, -1]-5.).relu() - wealth
    torch.testing.assert_close(result["terminal_loss"], expected_loss, atol=1e-9, rtol=0)
    torch.testing.assert_close(reward, -expected_loss.square(), atol=1e-8, rtol=0)
    assert torch.count_nonzero(result["liquidation_cost"]) == 0
    independent = finance.numpy_ledger(bank.marks.numpy(), targets.numpy(),
        finance.initial_state(bank).cash.numpy(), bank.liability[:, -1].numpy(), config)
    np.testing.assert_allclose(result["terminal_loss"], independent["terminal_loss"], atol=1e-14)
    # A different ending convention charges a real additional sell-out, not
    # another market movement. Calendar/instruments stay unchanged.
    closed_config = replace(config, settlement=SettlementConfig())
    closed = finance.ledger_from_positions(replace(bank, config=closed_config), targets)
    torch.testing.assert_close(closed["terminal_loss"]-result["terminal_loss"],
                               torch.full((2,), .05, dtype=torch.float64), atol=1e-14, rtol=0)


def test_source_search_keeps_maturity_trade_but_draws_no_fifth_market_move():
    config = maggiolo_gbm(step_days=365)
    game = SourceHedgingGame(config, seed=7, zeta=0., scale=1., grid_points=20)
    state = game.initial
    assert game.targets.shape == (20, 1)
    for _ in range(4):
        state, _ = game.getNextState(state, 1, 8)
    assert state.loss is None and state.date == 4
    before_rng = game.rng.bit_generator.state
    final, _ = game.getNextState(state, 1, 8)
    assert game.rng.bit_generator.state == before_rng
    torch.testing.assert_close(final.spot, state.spot, rtol=0, atol=0)
    assert final.loss is not None and final.date == 5


@dataclass(frozen=True)
class SquaredStockClaim:
    """An external instrument, without an engine-specific subclass or branch."""
    maturity: float
    kind: str = field(default="test_squared_stock", init=False)
    needs_integrated_variance = False

    @property
    def features(self):
        return dict(maturity=self.maturity)

    def mark(self, spot, variance, remaining, market, *, integrated_variance=None):
        # GBM, zero risk-neutral drift: E[S_T²|S_t] = S_t² exp(v*(T-t)).
        return spot.square() * torch.exp(variance*remaining)


def test_custom_instrument_works_as_hedge_and_liability_without_engine_changes():
    register_instrument("test_squared_stock", SquaredStockClaim)
    grid = TimeGrid(n_steps=3, step_days=2)
    custom = SquaredStockClaim(grid.horizon)
    config = HedgingConfig(GBMConfig(), grid,
        PortfolioConfig(custom, (custom,)), ExecutionConfig(), RiskConfig(objective="mse"))
    assert config_from_dict(asdict(config)) == config
    bank = finance.generate_market_bank(config, 5, 14, dtype=torch.float64)
    torch.testing.assert_close(bank.liability[:, -1], bank.spot[:, -1].square())
    env = TensorHedgingEnv(bank)
    for _ in range(config.n_decisions):
        targets = torch.tensor([[0., 1.]], dtype=torch.float64).expand(5, -1)
        obs, reward, done, _, result = env.step(targets)
    # Buy the identical claim once; settle it against the liability. Zero cost,
    # zero risk, independent of the path and of the chosen instrument's name.
    torch.testing.assert_close(result["terminal_loss"], torch.zeros(5, dtype=torch.float64), atol=1e-14, rtol=0)
    assert done and torch.isfinite(obs).all()
    # Custom claims may have negative values. Fees must never become rebates
    # simply because the instrument mark is negative.
    charged = replace(config, execution=ExecutionConfig(proportional=.01, quadratic=.02))
    fee = finance.transaction_cost(torch.tensor([[0., 2.]]), torch.tensor([[1., -3.]]), charged)
    torch.testing.assert_close(fee, torch.tensor([.30]))


def test_variance_integration_refinement_uses_internal_not_trading_steps():
    config = buehler_heston()
    fine = finance.simulate_market_paths(config.market, TimeGrid(8, 365), 3, 17,
                                         return_integrated_variance=True)
    coarse = finance.simulate_market_paths(config.market, TimeGrid(4, 365, step_days=2), 3, 17,
                                           substeps=2, return_integrated_variance=True)
    for actual, reference in zip(coarse, fine):
        torch.testing.assert_close(actual, reference[:, ::2], rtol=0, atol=0)


def test_source_named_profiles_keep_market_book_and_objective_together():
    for model in ("heston", "gbm"):
        config = szehr_market(model)
        assert config.n_assets == 1 and config.risk.objective == "mse"
        assert config_from_dict(asdict(config)) == config
