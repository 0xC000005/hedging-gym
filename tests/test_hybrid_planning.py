"""Mixed-gradient identity and causal planner/ledger integration."""
import itertools
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines.cem import RolloutPlanner
from hedging_gym.baselines.hpo import HybridPolicy, train_hybrid
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import TimeGrid
from hedging_gym.environment.finance import (
    generate_market_bank,
    market_bank_from_paths,
    numpy_ledger,
)
from hedging_gym.environment.gym_env import TensorHedgingEnv
from hedging_gym.environment.planning import rollout_branches, sample_continuation
from hedging_gym.environment.rollout import run_episode
from hedging_gym.evaluation import evaluate_controller


def test_hpo_mixed_gradient_matches_enumerated_expected_loss():
    torch.manual_seed(22)
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2))
    bank = generate_market_bank(config, 1, 51, dtype=torch.float64)
    policy = HybridPolicy(config, hidden=(4,)).double()
    exact, mixed = [], []
    # Sum over every joint mode history; no Monte Carlo tolerance is needed.
    for history in itertools.product(range(policy.n_modes), repeat=config.n_steps):
        env = TensorHedgingEnv(bank)
        observed = env.reset()
        log_scores = []
        for mode in history:
            distribution = Categorical(logits=policy.discrete(observed))
            log_scores.append(distribution.log_prob(torch.tensor([mode])))
            target = policy.candidates(observed, env.state.positions,
                config.execution.holding_lower, config.execution.holding_upper)[:, mode]
            observed, _, _, _, result = env.step(target)
        log_probability = torch.stack(log_scores).sum()
        probability = log_probability.exp()
        loss = result["terminal_loss"].square().sum()
        exact.append(probability * loss)
        mixed.append(probability.detach() * (loss + log_probability * loss.detach()))
    parameters = [*policy.discrete.parameters(), *policy.continuous.parameters()]
    direct = torch.autograd.grad(sum(exact), parameters, retain_graph=True)
    estimate = torch.autograd.grad(sum(mixed), parameters)
    for expected, actual in zip(direct, estimate):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    assert any(value.abs().max() > 1e-8 for value in direct)


def test_hybrid_training_and_search_preserve_frozen_weights_and_cash():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    # A legally fixed coordinate must not divide by zero during refinement.
    config = replace(config, execution=replace(config.execution,
                     holding_lower=(-1., 0.), holding_upper=(2., 0.)))
    train_bank = generate_market_bank(config, 16, 110, dtype=torch.float64)
    policy, metadata = train_hybrid(train_bank, updates=2, batch_size=8, hidden=(8,), progress=False)
    heldout = generate_market_bank(config, 4, 220, dtype=torch.float64)
    original = [parameter.detach().clone() for parameter in policy.parameters()]
    planner = RolloutPlanner(policy, zeta=metadata["zeta"], candidates=8, scenarios=4,
                             iterations=2, guided=True, gradient_steps=1)
    for controller in (policy_controller(policy), planner):
        metrics, tape = evaluate_controller(controller, heldout, batch_size=4)
        expected = numpy_ledger(heldout.marks.numpy(), tape["positions"].numpy(),
            heldout.liability[:, 0].numpy(), heldout.liability[:, -1].numpy(), config)
        np.testing.assert_allclose(tape["terminal_loss"], expected["terminal_loss"], atol=1e-11)
        assert metrics["constraint_violations"] == 0
    for before, after in zip(original, policy.parameters()):
        torch.testing.assert_close(before, after)
    assert planner.metadata()["conditional_rollout_paths"] > 0


def test_search_repeats_from_current_observation_not_realized_future():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2))
    bank = generate_market_bank(config, 3, 110)
    policy = HybridPolicy(config, hidden=(8,))
    environment = TensorHedgingEnv(bank)
    observed = environment.reset()
    # The planner receives no market bank, and two equal local RNG streams
    # yield the same action even if an unrelated future bank is generated.
    options = dict(zeta=.02, candidates=8, scenarios=4, iterations=2, seed=19)
    first = RolloutPlanner(policy, **options)(observed, environment.state, 0, config)
    generate_market_bank(config, 3, 990)
    second = RolloutPlanner(policy, **options)(observed, environment.state, 0, config)
    torch.testing.assert_close(first, second)
    assert torch.equal(environment.state.positions, torch.zeros_like(first))


@pytest.mark.parametrize("trade_at_maturity", [False, True])
def test_public_branches_match_episodes_and_action_gradients(trade_at_maturity):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3,
                              trade_at_maturity=trade_at_maturity))
    bank = generate_market_bank(config, 2, 11, dtype=torch.float64)
    env = TensorHedgingEnv(bank)
    paths = sample_continuation(env.reset(), 0, config, 3, torch.Generator().manual_seed(91))
    targets = torch.tensor([[[.2, .01], [.7, .02]], [[.3, .02], [.6, .01]]],
                           dtype=torch.float64, requires_grad=True)

    def hold(observed, ledger, time_index, config):
        return ledger.positions

    branched = rollout_branches(targets, env.state, 0, config, paths, hold)
    # Execute those same scenarios as ordinary episodes, one per candidate.
    def expand_states(coordinate):
        values = torch.stack([state[coordinate] for state in paths.states], dim=1)
        return values.reshape(2, 1, 3, -1).expand(-1, 2, -1, -1).reshape(12, -1)

    replay = market_bank_from_paths(config, expand_states(0), expand_states(1))

    def decide(observed, ledger, time_index, config):
        if time_index == 0:
            return targets[:, :, None].expand(-1, -1, 3, -1).reshape(12, 2)
        return ledger.positions

    linear = run_episode(decide, replay)["terminal_loss"].reshape(2, 2, 3)
    torch.testing.assert_close(branched, linear, rtol=0, atol=0)
    branch_gradient, = torch.autograd.grad(branched.sum(), targets, retain_graph=True)
    episode_gradient, = torch.autograd.grad(linear.sum(), targets)
    torch.testing.assert_close(branch_gradient, episode_gradient, rtol=0, atol=0)
    assert branch_gradient.abs().max() > 0
    assert torch.count_nonzero(env.state.positions) == 0

    # An external learner may reuse a float32 action buffer with a float64
    # ledger. Branch execution must own/cast actions just like ordinary steps.
    def buffered_controller():
        buffer = torch.empty(12, 2, dtype=torch.float32)

        def control(observed, ledger, time_index, config):
            if time_index == 0:
                return targets.detach().float()[:, :, None].expand(-1, -1, 3, -1).reshape(12, 2)
            return buffer.fill_(.05 * time_index)

        return control

    buffered_branch = rollout_branches(targets.detach().float(), env.state, 0, config,
                                       paths, buffered_controller())
    buffered_episode = run_episode(buffered_controller(), replay)["terminal_loss"].reshape(2, 2, 3)
    torch.testing.assert_close(buffered_branch, buffered_episode, rtol=0, atol=0)
