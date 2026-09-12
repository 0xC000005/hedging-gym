"""Exact two-date mixed-gradient identity, including live history cross terms."""
import itertools

import pytest
import torch

from hedging_gym.baselines.hpo import HybridPolicy
from hedging_gym.environment.benchmark import benchmark_config, operational_config
from hedging_gym.environment.config import TimeGrid
from hedging_gym.environment.finance import generate_market_bank
from hedging_gym.environment.gym_env import TensorHedgingEnv
from hedging_gym.extensions.joint_counterfactual import joint_objective


@pytest.mark.parametrize("score_scope", ["sampled_date", "trajectory"])
def test_joint_all_mode_and_sampled_gradients_match_exact_risk(score_scope):
    torch.manual_seed(23)
    config = operational_config(benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2)),
                                "operational_fixed")
    bank = generate_market_bank(config, 1, 81, dtype=torch.float64)
    policy = HybridPolicy(config, hidden=(4,)).double()
    policy.value.requires_grad_(False)
    parameters = list(policy.discrete.parameters()) + list(policy.continuous.parameters())
    boundaries = [set((0., 1.)), set((0., 1.))]
    exact_terms, detached_history_terms = [], []
    for first, second in itertools.product(range(policy.n_modes), repeat=2):
        env = TensorHedgingEnv(bank)
        observed = env.reset()
        probability, detached_probability = bank.spot.new_ones(1), bank.spot.new_ones(1)
        for date, mode in enumerate((first, second)):
            probabilities = policy.discrete(observed).softmax(-1)
            boundaries[date].update(probabilities.detach().cumsum(-1)[..., :-1].flatten().tolist())
            probability = probability * probabilities[:, mode]
            detached_probability = detached_probability * policy.discrete(observed.detach()).softmax(-1)[:, mode]
            target = policy.candidates(observed, env.state.positions,
                config.execution.holding_lower, config.execution.holding_upper)[:, mode]
            observed, _, _, _, result = env.step(target)
        exact_terms.append((probability, result["terminal_loss"]))
        detached_history_terms.append((detached_probability, result["terminal_loss"]))
    zeta = torch.stack([loss.detach() for _, loss in exact_terms]).mean()
    exact_objective = sum((probability * config.risk.loss(loss, zeta)).sum()
                          for probability, loss in exact_terms)
    exact = torch.autograd.grad(exact_objective, parameters, retain_graph=True)
    omitted_cross = torch.autograd.grad(sum((probability * config.risk.loss(loss, zeta)).sum()
        for probability, loss in detached_history_terms), parameters)
    continuous_start = len(list(policy.discrete.parameters()))
    assert max((a-b).abs().max() for a, b in zip(exact[continuous_start:],
                                                omitted_cross[continuous_start:])) > 1e-9
    intervals = [list(zip(sorted(values)[:-1], sorted(values)[1:])) for values in boundaries]
    for algorithm in ("all_mode", "sampled"):
        terms = []
        for (lo0, hi0), (lo1, hi1) in itertools.product(*intervals):
            weight = (hi0-lo0)*(hi1-lo1)
            if weight <= 0:
                continue
            uniforms = bank.spot.new_tensor([[(lo0+hi0)/2, (lo1+hi1)/2]])
            for date in range(2):
                objective, _, _ = joint_objective(policy, bank, time_index=date,
                    zeta=zeta, algorithm=algorithm, uniforms=uniforms, score_scope=score_scope)
                terms.append(.5 * weight * objective)
        estimated = torch.autograd.grad(sum(terms), parameters)
        for expected, actual in zip(exact, estimated):
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
        assert any(value.abs().max() > 1e-8 for value in estimated[:continuous_start])
        assert any(value.abs().max() > 1e-8 for value in estimated[continuous_start:])


def test_trajectory_all_mode_is_conditional_mean_of_sampled_gradient():
    torch.manual_seed(35)
    config = operational_config(benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2)),
                                "operational_fixed")
    bank = generate_market_bank(config, 1, 82, dtype=torch.float64)
    policy = HybridPolicy(config, hidden=(4,)).double()
    parameters = [*policy.discrete.parameters(), *policy.continuous.parameters()]
    zeta = bank.spot.new_tensor(.005)
    for date in range(2):
        uniforms = bank.spot.new_tensor([[.37, .73]])
        objective, branches, _ = joint_objective(policy, bank, time_index=date,
            zeta=zeta, uniforms=uniforms, score_scope="trajectory")
        integrated = torch.cat([x.flatten() for x in torch.autograd.grad(objective, parameters)])
        weights = branches["root_probabilities"][0].detach()
        boundaries = torch.cat((weights.new_zeros(1), weights.cumsum(0)))
        samples = []
        for mode in range(policy.n_modes):
            selected = uniforms.clone()
            selected[0, date] = (boundaries[mode]+boundaries[mode+1])/2
            objective, _, _ = joint_objective(policy, bank, time_index=date,
                zeta=zeta, algorithm="sampled", uniforms=selected, score_scope="trajectory")
            samples.append(torch.cat([x.flatten() for x in torch.autograd.grad(objective, parameters)]))
        samples = torch.stack(samples)
        expected = (weights[:, None]*samples).sum(0)
        torch.testing.assert_close(integrated, expected, rtol=1e-9, atol=1e-11)
        # There is genuine root-mode noise here, not an all-zero identity test.
        assert float((weights[:, None]*(samples-expected).square()).sum()) > 1e-10
