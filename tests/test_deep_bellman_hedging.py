"""Deep Bellman Hedging: utility equations, reward accounting, actor gradients and resume."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines.deep_bellman_hedging import (
    UTILITIES,
    DeepBellmanLearner,
    DeepBellmanPolicy,
    conditional_transition,
    entropic_certainty,
    make_controller,
    multistep_transition,
    observe,
    oce,
    sample_states,
    train_deep_bellman,
    transition,
)
from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import RiskConfig, SettlementConfig, TimeGrid
from hedging_gym.environment.finance import (
    generate_market_bank,
    initial_state,
    numpy_ledger,
    observation,
    observation_fields,
    trade_step,
)
from hedging_gym.environment.gym_env import TensorHedgingEnv
from hedging_gym.environment.paper_benchmarks import buehler_heston
from hedging_gym.environment.rollout import run_episode
from hedging_gym.evaluation import evaluate_controller


@pytest.fixture(scope="module")
def bank():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    return generate_market_bank(config, 24, 1101, dtype=torch.float64)


def _learner(config, utility="entropy", risk_aversion=.3):
    return DeepBellmanLearner(config, hidden=(8,), position_range=None, utility=utility,
                              risk_aversion=risk_aversion, learning_rate=1e-3, critic_learning_rate=1e-3,
                              device="cpu", dtype=torch.float64)


@pytest.mark.parametrize("name", sorted(UTILITIES))
def test_utilities_are_normalized_monotone_and_concave(name):
    utility, lam = UTILITIES[name], .7
    zero = torch.zeros((), dtype=torch.float64, requires_grad=True)
    value = utility(zero, lam)
    assert float(value.detach()) == 0.
    if name != "cvar":  # kinked at zero: left derivative 1 + lam, right derivative zero
        value.backward()
        assert float(zero.grad) == pytest.approx(1., abs=1e-12)
    values = utility(torch.linspace(-4., 4., 801, dtype=torch.float64), lam)
    assert (values.diff() >= -1e-12).all()
    assert (values.diff().diff() <= 1e-9).all()


def test_cvar_and_entropic_shifts_are_the_negative_threshold_losses():
    gains = torch.tensor([-2., 0., 3., .5], dtype=torch.float64)
    es, entropy = RiskConfig(alpha=.8), RiskConfig(objective="entropy", risk_aversion=.5)
    for shift in (-1., 0., 2.):
        shift = torch.tensor(shift, dtype=torch.float64)
        torch.testing.assert_close(oce(UTILITIES["cvar"], gains, shift, es.alpha/(1-es.alpha)),
                                   -es.loss(-gains, shift))
        torch.testing.assert_close(oce(UTILITIES["entropy"], gains, shift, entropy.risk_aversion),
                                   -entropy.loss(-gains, shift))
    shifts = torch.linspace(-5., 5., 200001, dtype=torch.float64)
    optimized = oce(UTILITIES["entropy"], gains[None], shifts[:, None], .5).mean(1).max()
    assert float(optimized) == pytest.approx(float(-entropy.entropic_risk(-gains)), abs=1e-6)
    mean = oce(UTILITIES["identity"], gains, torch.tensor(1.7, dtype=torch.float64), .5).mean()
    torch.testing.assert_close(mean, gains.mean())


@pytest.mark.parametrize("settlement", [SettlementConfig("liquidate", True), SettlementConfig("liquidate", False),
                                        SettlementConfig("mark_to_market")])
@pytest.mark.parametrize("inventory", [False, True])
@pytest.mark.parametrize("trade_at_maturity", [False, True])
def test_rewards_telescope_to_ledger_terminal_pnl(settlement, inventory, trade_at_maturity):
    config = replace(benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3, trade_at_maturity=trade_at_maturity)),
                     settlement=settlement)
    if inventory:
        config = replace(config, portfolio=replace(config.portfolio, initial_cash=2.5, initial_positions=(.3, -.2)))
    bank = generate_market_bank(config, 6, 91, dtype=torch.float64)
    torch.manual_seed(3)
    result = run_episode(policy_controller(DirectDHPolicy(config, hidden=(8,)).double()), bank,
                         record_positions=True)
    state = initial_state(bank)
    wealth = state.cash+(state.positions*bank.marks[:, 0]).sum(-1)-bank.liability[:, 0]
    rows, total = torch.arange(6), torch.zeros(6, dtype=torch.float64)
    for date in range(config.n_decisions):
        reward, _, alive = transition(bank, rows, torch.full((6,), date), state, result["positions"][:, date])
        total = total+reward
        assert alive.all() == (date < config.n_decisions-1)
        state = trade_step(state, result["positions"][:, date], bank.marks[:, min(date, config.n_steps)], config)
    torch.testing.assert_close(total, result["terminal_pnl"]-wealth, rtol=0, atol=1e-12)


def test_transition_reproduces_environment_observations(bank):
    config = bank.config
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    torch.manual_seed(5)
    rows = torch.arange(24)
    for date in range(config.n_decisions):
        dates = torch.full((24,), date)
        torch.testing.assert_close(observe(bank, rows, dates, env.state), observed, rtol=0, atol=0)
        targets = torch.rand(24, config.n_assets, dtype=torch.float64)*.4-.2
        reward, following, alive = transition(bank, rows, dates, env.state, targets)
        before = env.state.cash+(env.state.positions*bank.marks[:, date]).sum(-1)-bank.liability[:, date]
        observed, _, terminated, _, info = env.step(targets)
        if terminated:
            torch.testing.assert_close(reward, info["terminal_pnl"]-before, rtol=0, atol=0)
        else:
            torch.testing.assert_close(following, observed, rtol=0, atol=0)


def test_sampled_books_respect_bounds_and_actor_gradient_matches_finite_differences(bank):
    torch.manual_seed(11)
    learner = _learner(bank.config)
    with torch.no_grad():  # a nonzero critic, so the continuation value contributes
        learner.critic.eta.fill_(.5)
    rows, dates, state = sample_states(bank, 16, torch.Generator().manual_seed(2), learner.lower, learner.upper)
    assert ((state.positions >= learner.lower) & (state.positions <= learner.upper)).all()
    assert dates.min() >= 0 and dates.max() < bank.config.n_decisions
    observed = observe(bank, rows, dates, state)

    def objective():
        outcome, shift = learner.outcome(bank, rows, dates, state, observed)
        return oce(learner.utility, outcome, shift, learner.lam).mean()
    parameter = learner.policy.continuous[-1][-1].bias
    gradient, = torch.autograd.grad(objective(), parameter)
    index = int(gradient.abs().argmax())
    assert gradient[index].abs() > 1e-7
    epsilon = 1e-6
    with torch.no_grad():
        original = parameter[index].clone()
        parameter[index] = original+epsilon
        plus = objective()
        parameter[index] = original-epsilon
        minus = objective()
        parameter[index] = original
    torch.testing.assert_close(gradient[index], (plus-minus)/(2*epsilon), rtol=1e-5, atol=1e-9)


def test_buehler_zero_critic_starts_at_exactly_zero_and_then_moves(bank):
    torch.manual_seed(3)
    learner = _learner(bank.config)
    observed = observation(bank, 0, initial_state(bank))
    zeros = torch.zeros(24, 1, dtype=torch.float64)
    torch.testing.assert_close(learner.critic(observed), zeros, rtol=0, atol=0)
    rows, dates, state = sample_states(bank, 16, torch.Generator().manual_seed(2), learner.lower, learner.upper)
    learner.update(bank, rows, dates, state, actor_steps=1, critic_steps=1)
    assert not torch.equal(learner.critic(observed), zeros)


def test_unbounded_paper_book_requires_and_uses_position_range():
    config = buehler_heston()
    bank = generate_market_bank(config, 8, 17, dtype=torch.float64)
    with pytest.raises(ValueError, match="position_range"):
        train_deep_bellman(bank, updates=1, progress=False)
    policy, _ = train_deep_bellman(bank, seed=7, updates=2, batch_size=8, hidden=(8,),
                                   position_range=(2., 50.), progress=False)
    metrics, tape = evaluate_controller(make_controller(policy), bank)
    assert torch.isfinite(tape["positions"]).all() and np.isfinite(metrics["expected_shortfall"])


def test_training_changes_policy_and_frozen_evaluation_reconciles(bank):
    torch.manual_seed(7)
    initial = DeepBellmanPolicy(bank.config, hidden=(8,)).double()
    policy, metadata = train_deep_bellman(bank, seed=7, updates=3, batch_size=16, hidden=(8,), progress=False)
    assert any(not torch.equal(before, after) for before, after in zip(initial.parameters(), policy.parameters()))
    assert metadata["history"][-1]["completed"] == 3 and metadata["zeta"] is None
    assert metadata["options"]["utility"] == "cvar"
    assert metadata["options"]["risk_aversion"] == pytest.approx(bank.config.risk.alpha/(1-bank.config.risk.alpha))
    json.dumps(metadata, allow_nan=False)
    heldout = generate_market_bank(bank.config, 12, 2201, dtype=torch.float64)
    metrics, tape = evaluate_controller(policy_controller(policy), heldout, batch_size=5)
    assert np.isfinite(metrics["expected_shortfall"]) and metrics["constraint_violations"] == 0
    reference = numpy_ledger(heldout.marks.numpy(), tape["positions"].numpy(),
                             heldout.liability[:, 0].numpy(), heldout.liability[:, -1].numpy(), heldout.config)
    np.testing.assert_allclose(tape["terminal_loss"].numpy(), reference["terminal_loss"], atol=1e-12)


def test_mse_task_needs_an_explicit_utility_and_entropy_task_selects_entropy(bank):
    with pytest.raises(ValueError, match="monetary utility"):
        train_deep_bellman(replace(bank, config=replace(bank.config, risk=RiskConfig(objective="mse"))),
                           updates=1, progress=False)
    entropic = replace(bank, config=replace(bank.config, risk=RiskConfig(objective="entropy", risk_aversion=.2)))
    _, metadata = train_deep_bellman(entropic, updates=1, batch_size=8, hidden=(8,), progress=False)
    assert metadata["options"]["utility"] == "entropy" and metadata["options"]["risk_aversion"] == .2


def test_conditional_scenarios_repeat_rows_and_are_reproducible(bank):
    config = bank.config
    learner = _learner(config)
    rows, dates, state = sample_states(bank, 6, torch.Generator().manual_seed(2), learner.lower, learner.upper)
    targets = learner.targets(observe(bank, rows, dates, state), state.positions)
    draws = []
    for _ in range(2):
        reward, following, alive = conditional_transition(bank, rows, dates, state, targets, 3,
                                                          torch.Generator().manual_seed(11))
        draws.append(reward)
        assert reward.shape == (18,) and following.shape == (18, len(observation_fields(config)))
        assert torch.equal(alive, (dates != config.n_decisions-1).repeat_interleave(3))
        assert torch.isfinite(reward).all()
    torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)
    # Different scenarios of one state differ only through tomorrow's market.
    assert not torch.equal(draws[0][0], draws[0][1])
    with pytest.raises(ValueError, match="scenarios"):
        train_deep_bellman(replace(bank, config=replace(config, time_grid=replace(config.time_grid, trade_at_maturity=True))),
                           updates=1, scenarios=2, progress=False)


def test_entropic_aggregate_is_the_optimized_certainty_equivalent(bank):
    outcome = torch.tensor([[-2., 0., 3., .5], [1., 1., 1., 1.]], dtype=torch.float64)
    shifts = torch.linspace(-6., 6., 240001, dtype=torch.float64)
    optimized = oce(UTILITIES["entropy"], outcome[:, None, :], shifts[None, :, None], .5).mean(2).max(1).values
    torch.testing.assert_close(entropic_certainty(outcome, .5), optimized, atol=1e-6, rtol=0)
    torch.testing.assert_close(entropic_certainty(outcome, .5)[1], torch.tensor(1., dtype=torch.float64))
    with pytest.raises(ValueError, match="entropic"):
        train_deep_bellman(bank, updates=1, utility="entropy", aggregate="entropic", scenarios=1, progress=False)
    entropic = replace(bank, config=replace(bank.config, risk=RiskConfig(objective="entropy", risk_aversion=.2)))
    _, metadata = train_deep_bellman(entropic, updates=2, batch_size=8, hidden=(8,), scenarios=3,
                                     aggregate="entropic", progress=False)
    assert metadata["options"]["aggregate"] == "entropic" and metadata["history"][-1]["completed"] == 2


def test_multistep_rewards_telescope_and_stop_at_settlement(bank):
    config = bank.config
    learner = _learner(config)
    rows = torch.arange(24)
    state = initial_state(bank)
    with torch.no_grad():
        # From the first date with steps at the horizon, the target is the whole episode's P&L.
        reward, _, alive = multistep_transition(bank, rows, torch.zeros(24, dtype=torch.long), state,
                                                learner.targets(observe(bank, rows, torch.zeros(24, dtype=torch.long), state),
                                                                state.positions), config.n_decisions, learner.targets)
        episode = run_episode(policy_controller(learner.policy), bank)
    torch.testing.assert_close(reward, episode["terminal_pnl"], rtol=0, atol=1e-12)
    assert not alive.any()
    # Two steps from the second-to-last decision settle after one; from the first they bootstrap.
    dates = torch.tensor([config.n_decisions-2]*12+[0]*12)
    with torch.no_grad():
        observed = observe(bank, rows, dates, state)
        reward, following, alive = multistep_transition(bank, rows, dates, state,
                                                        learner.targets(observed, state.positions), 2, learner.targets)
    assert (~alive[:12]).all() and alive[12:].all()
    assert reward.shape == (24,) and torch.isfinite(reward).all() and following.shape[0] == 24
    with pytest.raises(ValueError, match="scenarios"):
        train_deep_bellman(bank, updates=1, scenarios=2, steps=2, progress=False)


@pytest.mark.parametrize("scenarios,steps", [(1, 1), (2, 1), (1, 3)])
def test_split_training_matches_uninterrupted(bank, tmp_path, scenarios, steps):
    options = dict(seed=7, batch_size=8, hidden=(8,), critic_steps=2, scenarios=scenarios, steps=steps, progress=False)
    full, full_meta = train_deep_bellman(bank, updates=4, **options)
    path = tmp_path/"latest.pt"
    train_deep_bellman(bank, updates=2, checkpoint_path=path, checkpoint_every=1, **options)
    resumed, meta = train_deep_bellman(bank, updates=4, resume_from=path, checkpoint_path=path, **options)
    for name, value in full.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        else:
            assert value == resumed.state_dict()[name]
    assert meta["history"][-1]["bellman_residual"] == full_meta["history"][-1]["bellman_residual"]
    assert (tmp_path/"latest-early.pt").exists()
    saved = torch.load(path, weights_only=False)
    assert saved["step"] == 4 and saved["learner"]["critic_optimizer"]["state"]
    with pytest.raises(ValueError, match="precede"):
        train_deep_bellman(bank, updates=1, resume_from=path, **options)
