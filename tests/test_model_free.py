"""Distribution equations, detached experience and common-ledger integration."""

from dataclasses import replace

import pytest
import torch

from hedging_gym.benchmark import benchmark_config
from hedging_gym.config import PortfolioConfig, TimeGrid
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import generate_market_bank
from methods.controllers import policy_controller
from methods.model_free import (DistributionalCritic, collect_episodes,
    _accumulated_cost, action_gradient_loss, gpd_expected_excess, gpd_nll, quantile_huber_loss, train_model_free)
from methods.policies import DirectDHPolicy


def test_quantile_loss_and_pareto_tail_equations():
    prediction = torch.zeros(1, 2, requires_grad=True)
    loss = quantile_huber_loss(prediction, torch.ones(1, 1), torch.tensor([.25, .75]))
    torch.testing.assert_close(loss, torch.tensor(.25))
    loss.backward()
    assert (prediction.grad < 0).all()
    # GPD(scale=1, shape=.5): survival=(1+.5*y)^-2 and mean=2.
    cutoff = torch.tensor([-1., 0., 3.], dtype=torch.float64)
    scale, shape = torch.ones(3, dtype=torch.float64), torch.full((3,), .5, dtype=torch.float64)
    torch.testing.assert_close(gpd_expected_excess(cutoff, scale, shape), cutoff.new_tensor([3., 2., .8]))
    torch.testing.assert_close(gpd_nll(cutoff.clamp_min(0), scale, shape), 3*torch.log1p(.5*cutoff.clamp_min(0)))


def test_pinball_loss_preserves_quantile_and_small_huber_approaches_it():
    # At q=9, 90% of this empirical distribution is below q; the 95% pinball
    # derivative must still move upward, independent of the return scale.
    target = torch.arange(10, dtype=torch.float64)[None]
    q = torch.tensor([[8.5]], dtype=torch.float64, requires_grad=True)
    probability = torch.tensor([.95], dtype=torch.float64)
    pinball = quantile_huber_loss(q, target, probability, 0)
    gradient, = torch.autograd.grad(pinball, q)
    torch.testing.assert_close(gradient, torch.tensor([[-.05]], dtype=torch.float64))
    for scale in (.01, 1., 100.):
        scaled = quantile_huber_loss(q*scale, target*scale, probability, .001*scale)/scale
        torch.testing.assert_close(scaled, pinball, atol=.0005, rtol=0)


def test_action_derivative_clipping_precedes_batch_average():
    action = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    values = (action*torch.tensor([[3., 4.], [0., .5]])).sum(-1)
    action_gradient_loss(values, action, clip=1.).backward()
    torch.testing.assert_close(action.grad, action.new_tensor([[.3, .4], [0., .25]]))


def test_exdrl_tail_is_used_for_targets_and_actor_gradients():
    critic = DistributionalCritic(3, 2, hidden=(8,), quantiles=32, tail_threshold=.8).double()
    observed, actions = torch.zeros(4, 3).double(), torch.zeros(4, 2).double()
    raw, _, _ = critic(observed, actions)
    spliced = critic.distribution(observed, actions)
    mask = critic.probabilities > .8
    assert not torch.equal(spliced[:, mask], raw.sort(-1).values[:, mask])
    objective = critic.expected_ru(observed, actions, torch.tensor(0.), .95).mean()
    objective.backward()
    assert critic.tail[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(critic.tail_loss(observed, actions))


def test_ru_threshold_gradient_keeps_money_units_after_critic_scaling():
    critic = DistributionalCritic(1, 1, hidden=(), quantiles=4).double()
    with torch.no_grad():
        critic.quantiles[-1].weight.zero_()
        critic.quantiles[-1].bias.copy_(torch.tensor([-2., -1., 1., 3.]))
    observed, action = torch.zeros(1, 1).double(), torch.zeros(1, 1).double()
    for scale in (.001, 1., 1000.):
        threshold_money = torch.tensor(.5*scale, dtype=torch.float64, requires_grad=True)
        objective_money = scale * critic.expected_ru(observed, action, threshold_money/scale, .8).mean()
        gradient, = torch.autograd.grad(objective_money, threshold_money)
        # Two of four atoms exceed the threshold: dRU/dzeta = 1 - .5/.2.
        torch.testing.assert_close(gradient, gradient.new_tensor(-1.5))


@pytest.mark.parametrize("method,stock_only", [("hull_rl", True), ("exdrl", False)])
def test_model_free_updates_detached_environment_and_common_evaluation(method, stock_only):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    if stock_only:
        config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3),
            portfolio=PortfolioConfig(config.portfolio.liability, ()))
    bank = generate_market_bank(config, 24, 811, dtype=torch.float64)
    torch.manual_seed(7)
    initial = DirectDHPolicy(config, hidden=(8,)).double()
    transitions, losses, _ = collect_episodes(initial, replace(bank, marks=bank.marks.requires_grad_()), n_step=2)
    assert all(not value.requires_grad for value in transitions)
    assert not losses.requires_grad
    # At the first of three dates, a two-step transition must bootstrap;
    # the other two dates carry the actual terminal accounting loss.
    costs, done = transitions[2].reshape(24, 3), transitions[4].reshape(24, 3)
    assert not done[:, 0].any() and done[:, 1:].all()
    torch.testing.assert_close(costs[:, 0], torch.zeros(24, dtype=torch.float64))
    torch.testing.assert_close(costs[:, 1:], losses[:, None].expand(-1, 2))
    policy, metadata = train_model_free(method, bank, seed=7, updates=3, batch_size=8,
        hidden=(8,), quantiles=32, tail_threshold=.8, gradient_steps=2, progress=False)
    assert any(not torch.equal(before, after) for before, after in zip(initial.parameters(), policy.parameters()))
    assert metadata["gradient_updates"] == 6
    assert metadata["history"][-1]["completed"] == 3
    heldout = generate_market_bank(config, 12, 822, dtype=torch.float64)
    metrics, tape = evaluate_controller(policy_controller(policy), heldout, zeta=metadata["zeta"])
    assert metrics["constraint_violations"] == 0
    assert torch.isfinite(tape["terminal_loss"]).all()
    assert policy.n_assets == config.n_assets


@pytest.mark.parametrize("method", ["hull_rl", "exdrl"])
def test_model_free_checkpoint_resume_matches_uninterrupted_training(tmp_path, method):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 24, 811, dtype=torch.float64)
    options = dict(seed=7, batch_size=8, hidden=(8,), quantiles=32,
                   tail_threshold=.8, gradient_steps=2, replay_capacity=256, progress=False,
                   dense_rewards=method == "exdrl", actor_warmup_updates=1,
                   actor_update_period=3, action_gradient_clip=1., quantile_kappa=.01,
                   tail_learning_rate=1e-6, tail_policy_actions=True)
    full_path, split_path = tmp_path/"full.pt", tmp_path/"split.pt"
    full, _ = train_model_free(method, bank, updates=4, checkpoint_path=full_path, **options)
    train_model_free(method, bank, updates=2, checkpoint_path=split_path, **options)
    resumed, _ = train_model_free(method, bank, updates=4, checkpoint_path=split_path,
                                  resume_from=split_path, **options)
    for actual, expected in zip(resumed.parameters(), full.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual = torch.load(split_path, weights_only=False)
    expected = torch.load(full_path, weights_only=False)
    for name in ("critic", "target_critic", "target_actor"):
        for key, tensor in actual[name].items():
            if isinstance(tensor, torch.Tensor):
                torch.testing.assert_close(tensor, expected[name][key], rtol=0, atol=0)
    assert actual["replay"]["cursor"] == expected["replay"]["cursor"]
    assert (tmp_path/"split-early.pt").exists()


def test_critic_warmup_freezes_actor_but_updates_critic(tmp_path):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 24, 811, dtype=torch.float64)
    torch.manual_seed(7)
    initial = DirectDHPolicy(config, hidden=(8,)).double()
    policy, metadata = train_model_free("exdrl", bank, seed=7, updates=2, batch_size=8,
        hidden=(8,), quantiles=32, tail_threshold=.8, gradient_steps=2,
        actor_warmup_updates=2, tail_policy_actions=True, tail_learning_rate=1e-6,
        checkpoint_path=tmp_path/"warmup.pt", progress=False)
    for actual, expected in zip(policy.parameters(), initial.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    saved = torch.load(tmp_path/"warmup.pt", weights_only=False)
    assert saved["critic_optimizer"]["state"]
    assert not saved["actor_optimizer"]["state"]
    assert metadata["actor_gradient_updates"] == 0


def test_explicit_frozen_warmup_extension_matches_declared_longer_run(tmp_path):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 24, 811, dtype=torch.float64)
    options = dict(seed=7, batch_size=8, hidden=(8,), quantiles=32, tail_threshold=.8,
                   gradient_steps=2, progress=False)
    full, _ = train_model_free("exdrl", bank, updates=4, actor_warmup_updates=3, **options)
    path = tmp_path/"frozen.pt"
    train_model_free("exdrl", bank, updates=2, actor_warmup_updates=2, checkpoint_path=path, **options)
    with pytest.raises(ValueError, match="saved training options"):
        train_model_free("exdrl", bank, updates=4, actor_warmup_updates=3, resume_from=path, **options)
    resumed, _ = train_model_free("exdrl", bank, updates=4, actor_warmup_updates=3,
        resume_from=path, checkpoint_path=path, extend_frozen_warmup=True, **options)
    for actual, expected in zip(resumed.parameters(), full.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="never updated"):
        train_model_free("exdrl", bank, updates=5, actor_warmup_updates=5,
            resume_from=path, extend_frozen_warmup=True, **options)


def test_dense_rewards_preserve_terminal_loss_and_global_ru():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 24, 811, dtype=torch.float64)
    actor = DirectDHPolicy(config, hidden=(8,)).double()
    transitions, losses, _ = collect_episodes(actor, bank, n_step=1, dense_rewards=True)
    torch.testing.assert_close(transitions[2].reshape(24,3).sum(-1), losses)
    complete, _, _ = collect_episodes(actor, bank, n_step=3, dense_rewards=True)
    accumulated = _accumulated_cost(complete[0], actor, config, bank.liability[0,0])
    # L = accumulated + remaining. The transformed threshold is zeta-accumulated;
    # never optimize a separate conditional ES at each date.
    remaining = complete[2]
    threshold = losses.new_tensor(.015, requires_grad=True)
    transformed = accumulated + config.risk.loss(remaining, threshold-accumulated)
    original = config.risk.loss(losses[:,None].expand(-1,3).flatten(),threshold)
    torch.testing.assert_close(transformed,original)
    torch.testing.assert_close(torch.autograd.grad(transformed.mean(),threshold,retain_graph=True)[0],
                               torch.autograd.grad(original.mean(),threshold)[0])
