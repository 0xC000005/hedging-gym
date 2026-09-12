"""Chance semantics, source learning loop and common-ledger integration."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym.baselines.alphazero import (
    AlphaZeroPolicy,
    _completed_rollouts,
    _FinanceSearch,
    alphazero_controller,
    completed_rollout_action_diagnostic,
    stochastic_search_batch,
    train_alphazero,
)
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import EuropeanOption, TimeGrid
from hedging_gym.environment.finance import (
    generate_market_bank,
    liquidate,
    mark_state,
    numpy_ledger,
    observation,
    trade_step,
)
from hedging_gym.environment.gym_env import TensorHedgingEnv
from hedging_gym.evaluation import evaluate_controller


def test_chance_outcomes_are_averaged_and_tree_reaches_terminal_states():
    class CoinModel:
        draws = 0

        def terminal_cost(self, state):
            return state

        def evaluate(self, states):
            return [(np.ones(1), 1.) for _ in states]

        def advance(self, states, actions, rngs):
            output = []
            for _ in states:
                # Exactly balanced sampled outcomes: the search must average
                # rather than treat the favorable zero-cost branch as a move.
                output.append(float(2*(self.draws % 2)))
                self.draws += 1
            return output

    result, = stochastic_search_batch([None], CoinModel(), [np.random.default_rng(9)], simulations=1024)
    assert .7 < result["value"] < 1.3
    assert result["work"]["terminal_evaluations"] > 1

    class ThreeStepModel:
        def terminal_cost(self, state):
            depth, cost = state
            return float(cost) if depth == 3 else None

        def evaluate(self, states):
            return [(np.full(2, .5), float(cost)+(3-depth)*.5) for depth, cost in states]

        def advance(self, states, actions, rngs):
            return [(depth+1, cost+action) for (depth, cost), action in zip(states, actions)]

    result, = stochastic_search_batch([(0, 0)], ThreeStepModel(), [np.random.default_rng(3)],
                                      simulations=256)
    assert result["work"]["maximum_depth"] == 3
    assert result["work"]["terminal_evaluations"] > 0
    assert result["policy"][0] > result["policy"][1]

    # Batching independent roots must not delay an individual tree's backups
    # or mix its random stream with another root.
    seeds = (3, 7, 11)
    batched = stochastic_search_batch([(0, 0)]*len(seeds), ThreeStepModel(),
        [np.random.default_rng(seed) for seed in seeds], simulations=64)
    for seed, together in zip(seeds, batched):
        alone, = stochastic_search_batch([(0, 0)], ThreeStepModel(),
            [np.random.default_rng(seed)], simulations=64)
        np.testing.assert_array_equal(together["visits"], alone["visits"])
        assert together["value"] == alone["value"]
        assert together["work"] == alone["work"]


@pytest.mark.parametrize("extra_option", [False, True])
def test_search_respects_instrument_geometry_constraints_and_ledger(extra_option):
    config = benchmark_config(model="heston", time_grid=TimeGrid(n_steps=3))
    if extra_option:
        config = replace(config, portfolio=replace(config.portfolio,
            hedges=(*config.portfolio.hedges, EuropeanOption(1.1, 8*config.dt, "put"))),
            execution=replace(config.execution, holding_lower=-1., holding_upper=1., proportional=.001))
    config = replace(config, execution=replace(config.execution, holding_lower=-.3, holding_upper=.3,
                                              minimum_trade=.2, trade_lot=.1))
    bank = generate_market_bank(config, 4, 51, dtype=torch.float64)
    policy = AlphaZeroPolicy(config, hidden=(8,), grid_points=3).double()
    controller = alphazero_controller(policy, simulations=8, seed=16)
    env = TensorHedgingEnv(bank)
    observation = env.reset()
    original = env.state.positions.clone()
    _, legal = policy.candidates(env.state.positions, config)
    assert legal[:, :-1].any(-1).all()  # Decimal lots must not collapse the grid to HOLD.
    target = controller(observation, env.state, 0, config)
    torch.testing.assert_close(env.state.positions, original)
    assert target.shape == (4, config.n_assets)
    metrics, tape = evaluate_controller(alphazero_controller(policy, simulations=8, seed=16), bank)
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(), bank.liability[:, 0].numpy(),
                             bank.liability[:, -1].numpy(), config)
    np.testing.assert_allclose(tape["terminal_loss"].numpy(), reference["terminal_loss"], atol=1e-11)
    assert metrics["constraint_violations"] == 0


def test_self_play_fits_policy_and_value_with_shared_terminal_risk():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 16, 103, dtype=torch.float64)
    torch.manual_seed(7)
    initial = AlphaZeroPolicy(config, hidden=(8,)).double()
    policy, metadata = train_alphazero(bank, updates=2, batch_size=4, hidden=(8,),
        simulations=8, gradient_steps=2, calibration_paths=16, reanalysis_states=8,
        reanalysis_samples=8, value_gradient_steps=8, progress=False)
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), policy.parameters()))
    assert metadata["optimizer_steps"] == 4
    assert metadata["work"]["terminal_evaluations"] > 0
    assert metadata["history"][-1]["completed"] == 2
    assert all(np.isfinite(record["value_loss"]) for record in metadata["history"])
    heldout = generate_market_bank(config, 8, 203, dtype=torch.float64)
    metrics, _ = evaluate_controller(alphazero_controller(policy, simulations=8), heldout,
                                    zeta=metadata["zeta"])
    assert np.isfinite(metrics["ru_at_training_zeta"])

    # Direct one-step adapter result agrees with the environment at settlement,
    # including a short put liability and transaction/liquidation costs.
    changed = replace(config, portfolio=replace(config.portfolio, liability_quantity=-1.5,
                                              liability=replace(config.portfolio.liability, kind="put")))
    changed_bank = generate_market_bank(changed, 1, 204, dtype=torch.float64)
    env = TensorHedgingEnv(changed_bank)
    obs = env.reset()
    for _ in range(changed.n_steps-1):
        obs, *_ = env.step(torch.zeros_like(env.state.positions))
    model = _FinanceSearch(policy, changed)
    root, = model.roots(obs, env.state, changed.n_steps-1)
    child, = model.advance([root], [len(policy.targets)], [np.random.default_rng(40)])
    _, payoff = mark_state(child.spot, child.variance, changed.n_steps, changed)
    loss = liquidate(child.ledger, child.marks, payoff, changed)["terminal_loss"]
    assert child.terminal_cost == pytest.approx(float(changed.risk.loss(loss, policy.zeta)[0]))
    assert child.terminal_loss == pytest.approx(float(loss[0]))


def test_checkpoint_resume_preserves_search_replay_and_training(tmp_path):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 16, 103, dtype=torch.float64)
    options = dict(batch_size=4, hidden=(8,), simulations=8, gradient_steps=2, progress=False,
        calibration_paths=16, reanalysis_states=8, reanalysis_samples=4, value_gradient_steps=8)
    complete, complete_metadata = train_alphazero(bank, updates=3, **options)
    checkpoint = tmp_path / "alphazero.pt"
    train_alphazero(bank, updates=1, checkpoint_path=checkpoint, **options)
    resumed, resumed_metadata = train_alphazero(bank, updates=3, resume_from=checkpoint,
                                               checkpoint_path=checkpoint, **options)
    for expected, actual in zip(complete.parameters(), resumed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(resumed.zeta, complete.zeta, rtol=0, atol=0)
    assert resumed_metadata["work"] == complete_metadata["work"]
    assert resumed_metadata["history"][-1]["value_loss"] == complete_metadata["history"][-1]["value_loss"]
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["step"] == 3 and len(saved["replay"]) == 3
    assert checkpoint.with_name("alphazero-early.pt").exists()
    torch.testing.assert_close(resumed.value_scale, complete.value_scale, rtol=0, atol=0)
    assert resumed_metadata["refit_work"] == complete_metadata["refit_work"]
    assert saved["value_replay"][1].shape == (8, 4)


def test_checkpoint_threshold_matches_frozen_actor_and_raw_returns_are_relabelable(tmp_path):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 64, 991, dtype=torch.float64)
    checkpoint = tmp_path / "calibrated.pt"
    policy, _ = train_alphazero(bank, updates=1, batch_size=8, hidden=(8,),
        simulations=8, gradient_steps=2, calibration_paths=64, reanalysis_states=8,
        reanalysis_samples=8, value_gradient_steps=8, checkpoint_path=checkpoint, progress=False)
    _, tape = evaluate_controller(alphazero_controller(policy, simulations=0), bank)
    torch.testing.assert_close(policy.zeta, torch.quantile(tape["terminal_loss"], config.risk.alpha))
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    logits, _ = policy(policy.features(observed, spot0=config.market.spot0))
    saved = torch.load(checkpoint, weights_only=False)
    raw = saved["value_replay"][1]
    assert (raw < policy.zeta).any()  # Below-threshold returns remain uncensored.
    low_zeta = raw.min() - .01
    assert (config.risk.loss(raw, low_zeta) > low_zeta).all()
    policy.zeta.add_(.5)
    after, _ = policy(policy.features(observed, spot0=config.market.spot0))
    torch.testing.assert_close(logits, after, rtol=0, atol=0)


def test_completed_action_diagnostic_uses_exact_terminal_cost_at_one_step():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=1))
    policy = AlphaZeroPolicy(config, hidden=(8,)).double()
    result, tape = completed_rollout_action_diagnostic(policy, config, samples=8, seed=99)
    expected = config.risk.loss(tape["terminal_loss"], policy.zeta).mean(-1)
    torch.testing.assert_close(torch.tensor(result["completed_ru"]), expected, check_dtype=False)
    assert result["mean_absolute_value_error"] < 1e-12
    assert not result["action_ordering_screen_failed"]
    assert result["work"]["terminal_evaluations"] == len(result["action_indices"])*8


def test_reanalysis_restores_paid_costs_through_cash_and_completes_same_policy():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 4, 1201, dtype=torch.float64)
    policy = AlphaZeroPolicy(config, hidden=(8,)).double()
    env = TensorHedgingEnv(bank)
    before = policy.value_features(policy.features(env.reset(), spot0=config.market.spot0))[:, -1]
    traded = trade_step(env.state, policy.targets[0].expand(4, -1), bank.marks[:, 0], config)
    after = policy.value_features(policy.features(observation(bank, 0, traded),
                                                  spot0=config.market.spot0))[:, -1]
    torch.testing.assert_close((before-after)*policy.search_scale, traded.total_cost,
                               atol=1e-14, rtol=1e-12)
    observed, *_ = env.step(policy.targets[0].expand(4, -1))
    assert (env.state.total_cost > 0).all()
    model = _FinanceSearch(policy, config)
    original = model.roots(observed, env.state, 1)
    restored = model.observed_roots(observed)
    first, _, work = _completed_rollouts(model, original, [17, 19, 23, 29])
    second, _, restored_work = _completed_rollouts(model, restored, [17, 19, 23, 29])
    np.testing.assert_array_equal(first, second)
    assert restored_work == work
    assert work["terminal_evaluations"] == 4
    assert work["transition_samples"] == 8


def test_repeating_an_absolute_grid_action_remains_a_hold():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    policy = AlphaZeroPolicy(config, hidden=(8,))
    # Source actions represent absolute holdings, so repeating one must not
    # force a switch to the second-best action just to canonicalize HOLD.
    target = policy.targets[0].float()[None]
    candidates, legal = policy.candidates(target, config)
    assert legal[0, 0] and not legal[0, -1]
    torch.testing.assert_close(candidates[0, 0], target[0])
    off_grid = torch.full_like(target, .123)
    _, off_grid_legal = policy.candidates(off_grid, config)
    assert off_grid_legal[0, -1]
