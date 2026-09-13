"""Focused adapter checks: gradients, updates and authoritative episode tapes."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines._shared.training import rollout
from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.baselines.deep_hedging import train as train_dh
from hedging_gym.baselines.delta import make_controller as delta_controller
from hedging_gym.baselines.delta import spot_delta
from hedging_gym.baselines.delta_gamma import make_controller as delta_gamma_controller
from hedging_gym.baselines.delta_gamma import spot_gamma_greeks
from hedging_gym.baselines.delta_variance import (
    make_controller as delta_variance_controller,
)
from hedging_gym.baselines.delta_variance import spot_variance_greeks
from hedging_gym.baselines.no_transaction_band import NoTransactionBandPolicy
from hedging_gym.baselines.no_transaction_band import train as train_ntb
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import (
    EuropeanOption,
    ExecutionConfig,
    GBMConfig,
    HedgingConfig,
    PortfolioConfig,
    RiskConfig,
    TimeGrid,
)
from hedging_gym.environment.finance import (
    generate_market_bank,
    instrument_names,
    numpy_ledger,
    observation_fields,
)
from hedging_gym.environment.gym_env import HedgingVectorEnv
from hedging_gym.environment.rollout import run_episode
from hedging_gym.evaluation import evaluate_controller


@pytest.fixture(scope="module")
def bank():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    return generate_market_bank(config, 24, 1101, dtype=torch.float64)


@pytest.mark.parametrize("policy_class", [DirectDHPolicy, NoTransactionBandPolicy])
def test_policy_terminal_gradient_and_evaluator_tape(bank, policy_class):
    torch.manual_seed(19)
    policy = policy_class(bank.config, hidden=(8,)).double()
    output = run_episode(policy_controller(policy, evaluation=False), bank, record_positions=True)
    assert policy.training  # The common runner must not switch off training mode.
    objective = output["terminal_loss"].mean()
    parameter = policy.continuous[-1].bias
    gradient, = torch.autograd.grad(objective, parameter)
    assert torch.isfinite(gradient).all()
    index = int(gradient.abs().argmax())
    assert gradient[index].abs() > 1e-7
    epsilon = 1e-5
    with torch.no_grad():
        original = parameter[index].clone()
        parameter[index] = original+epsilon
        plus = rollout(policy, bank)["terminal_loss"].mean()
        parameter[index] = original-epsilon
        minus = rollout(policy, bank)["terminal_loss"].mean()
        parameter[index] = original
    torch.testing.assert_close(gradient[index], (plus-minus)/(2*epsilon), rtol=2e-4, atol=1e-8)
    # Shared training/evaluation episodes must execute the same actual trades.
    metrics, tape = evaluate_controller(policy_controller(policy), bank, batch_size=7)
    torch.testing.assert_close(tape["positions"], output["positions"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(tape["terminal_loss"], output["terminal_loss"], rtol=1e-10, atol=1e-12)
    assert metrics["constraint_violations"] == 0
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
                             bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
    for key in ("terminal_loss", "transaction_cost", "turnover", "tickets"):
        np.testing.assert_allclose(tape[key].numpy(), reference[key], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method,policy_class", [("dh", DirectDHPolicy), ("ntb", NoTransactionBandPolicy)])
def test_training_changes_policy_and_frozen_evaluation_is_finite(bank, method, policy_class):
    torch.manual_seed(7)
    initial = policy_class(bank.config, hidden=(8,)).double()
    policy, metadata = {'dh': train_dh, 'ntb': train_ntb}[method](bank, seed=7, updates=3, batch_size=16,
                                    hidden=(8,), progress=False)
    assert any(not torch.equal(before, after) for before, after in zip(initial.parameters(), policy.parameters()))
    assert metadata["history"][-1]["completed"] == 3
    heldout = generate_market_bank(bank.config, 12, 2201, dtype=torch.float64)
    metrics, _ = evaluate_controller(policy_controller(policy), heldout, zeta=metadata["zeta"])
    assert np.isfinite(metrics["expected_shortfall"])
    assert np.isfinite(metrics["ru_at_training_zeta"])


def test_entropic_objective_trains_and_reports_pooled_entropic_risk():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3),
                              risk=RiskConfig(objective="entropy", risk_aversion=.1))
    bank = generate_market_bank(config, 24, 1101, dtype=torch.float64)
    policy, metadata = train_dh(bank, seed=7, updates=3, batch_size=16, hidden=(8,), progress=False)
    assert metadata["history"][-1]["completed"] == 3
    heldout = generate_market_bank(config, 12, 2201, dtype=torch.float64)
    metrics, tape = evaluate_controller(policy_controller(policy), heldout, zeta=metadata["zeta"])
    assert metrics["objective"] == "entropy"
    entropic = float(config.risk.entropic_risk(tape["terminal_loss"].double()))
    assert metrics["objective_value"] == pytest.approx(entropic, abs=1e-12)
    assert float(config.risk.loss(tape["terminal_loss"].double(), metadata["zeta"]).mean()) >= entropic
    assert metrics["ru_at_training_zeta"] == pytest.approx(
        float(config.risk.loss(tape["terminal_loss"].double(), metadata["zeta"]).mean()), abs=1e-12)


@pytest.mark.parametrize("method", ["delta", "delta_gamma", "delta_variance"])
def test_classical_controller_uses_common_evaluation_accounting(bank, method):
    metrics, tape = evaluate_controller({'delta': delta_controller, 'delta_gamma': delta_gamma_controller, 'delta_variance': delta_variance_controller}[method](), bank, batch_size=7)
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
                             bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
    np.testing.assert_allclose(tape["terminal_loss"].numpy(), reference["terminal_loss"], atol=1e-12)
    assert metrics["constraint_violations"] == 0


def test_continuous_training_does_not_silently_project_lot_constraints(bank):
    constrained = replace(bank, config=replace(bank.config,
        execution=replace(bank.config.execution, trade_lot=.1)))
    with pytest.raises(ValueError, match="minimum-trade or lot"):
        train_dh(constrained, updates=1, progress=False)


def _book(*, stock_only=False, quantity=1., liability_kind="put", alpha=.8):
    clock = TimeGrid(n_steps=3, days_per_year=365)
    return HedgingConfig(
        market=GBMConfig(spot0=100., v0=.09), time_grid=clock,
        portfolio=PortfolioConfig(EuropeanOption(100., clock.horizon, liability_kind),
            () if stock_only else (EuropeanOption(90., 8*clock.dt),
                                   EuropeanOption(105., 10*clock.dt, "put")), quantity),
        execution=ExecutionConfig(proportional=.001, holding_lower=-20., holding_upper=20.),
        risk=RiskConfig(alpha=alpha),
    )


@pytest.mark.parametrize("policy_class", [DirectDHPolicy, NoTransactionBandPolicy])
@pytest.mark.parametrize("stock_only", [True, False])
def test_policy_from_env_derives_stock_only_and_mixed_option_geometry(policy_class, stock_only):
    config = _book(stock_only=stock_only, quantity=-1.5)
    env = HedgingVectorEnv(5, config)
    try:
        policy = policy_class.from_env(env, hidden=(8,))
        observed, _ = env.reset_tensor(seed=120)
        controller = policy_controller(policy)
        for date in range(config.n_steps):
            target = controller(observed, env._tensor_env.state, date, config)
            assert target.shape == (5, config.n_assets)
            observed, _, terminated, _, info = env.step_tensor(target)
        assert terminated.all() and torch.isfinite(info["terminal_loss"]).all()
        assert policy.observation_fields == tuple(observation_fields(config))
        assert policy.instrument_names == tuple(instrument_names(config))
    finally:
        env.close()


@pytest.mark.parametrize("policy_class", [DirectDHPolicy, NoTransactionBandPolicy])
def test_policy_schema_accepts_parameter_change_but_rejects_same_width_option_change(policy_class):
    config = _book()
    policy = policy_class(config, hidden=(8,)).double()
    changed_market = replace(config, market=replace(config.market, v0=.16, mu=.03))
    policy.check_config(changed_market)
    changed_bank = generate_market_bank(changed_market, 3, 222, dtype=torch.float64)
    assert torch.isfinite(rollout(policy, changed_bank)["terminal_loss"]).all()

    changed_portfolio = replace(config.portfolio,
        hedges=(replace(config.portfolio.hedges[0], kind="put"), config.portfolio.hedges[1]))
    changed_contract = replace(config, portfolio=changed_portfolio)
    assert len(observation_fields(changed_contract)) == policy.feature_dim
    with pytest.raises(ValueError, match="schema"):
        policy.check_config(changed_contract)
    incompatible = policy_class(changed_contract, hidden=(8,)).double()
    with pytest.raises(ValueError, match="schema"):
        incompatible.load_state_dict(policy.state_dict())
    compatible = policy_class(config, hidden=(8,)).double()
    compatible.load_state_dict(policy.state_dict())
    for before, restored in zip(policy.parameters(), compatible.parameters()):
        torch.testing.assert_close(before, restored)


@pytest.mark.parametrize("quantity", [2.5, -1.5])
def test_classical_put_liability_quantity_and_selected_put_hedge(quantity):
    config = _book(quantity=quantity)
    spot = torch.tensor([97., 100., 103.], dtype=torch.float64)
    variance = torch.full_like(spot, config.market.v0)
    put_delta = spot_delta(spot, variance, 0, config)
    call_config = replace(config, portfolio=replace(config.portfolio,
        liability=replace(config.portfolio.liability, kind="call")))
    torch.testing.assert_close(put_delta, spot_delta(spot, variance, 0, call_config)-quantity)
    bank = generate_market_bank(config, 3, 1103, dtype=torch.float64)
    for method, greek in (("delta_gamma", spot_gamma_greeks),
                          ("delta_variance", spot_variance_greeks)):
        _, tape = evaluate_controller({'delta': delta_controller, 'delta_gamma': delta_gamma_controller, 'delta_variance': delta_variance_controller}[method](hedge_index=1), bank)
        target = tape["positions"][:, 0]
        ds, sensitivity = greek(bank.spot[:, 0], bank.variance[:, 0], 0, config, hedge_index=1)
        assert (target[:, 1] == 0).all()
        torch.testing.assert_close(target[:, 2]*sensitivity[:, 1], sensitivity[:, 0])
        torch.testing.assert_close(target[:, 0]+target[:, 2]*ds[:, 1], ds[:, 0])


def test_stock_only_delta_uses_signed_put_liability():
    config = _book(stock_only=True, quantity=2.)
    bank = generate_market_bank(config, 4, 441, dtype=torch.float64)
    _, tape = evaluate_controller(delta_controller(), bank)
    assert tape["positions"].shape == (4, config.n_steps, 1)
    assert (tape["positions"] <= 0).all()
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), config)
    np.testing.assert_allclose(tape["terminal_loss"].numpy(), reference["terminal_loss"], atol=1e-11)
    with pytest.raises(ValueError, match="available hedge option"):
        evaluate_controller(delta_gamma_controller(), bank)


@pytest.mark.parametrize("method", ["dh", "ntb"])
def test_training_initial_threshold_uses_configured_risk_confidence(method):
    config = _book(stock_only=True, alpha=.5)
    bank = generate_market_bank(config, 24, 3301, dtype=torch.float64)
    # Negligible optimizer movement leaves the independently observed initial
    # median unchanged, exposing an accidentally hardcoded tail confidence.
    policy, metadata = {'dh': train_dh, 'ntb': train_ntb}[method](bank, seed=19, updates=1, batch_size=16,
        hidden=(8,), learning_rate=1e-30, zeta_learning_rate=1e-30, progress=False)
    initial_losses = rollout(policy, bank)["terminal_loss"].detach()
    assert metadata["zeta"] == pytest.approx(float(torch.quantile(initial_losses, .5)), abs=1e-12)
    assert metadata["history"][0]["risk_alpha"] == .5
