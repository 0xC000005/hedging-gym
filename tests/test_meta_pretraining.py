"""Exact tail gradients, first-order mapping and source-only resumability."""

from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn

from hedging_gym.benchmark import benchmark_config
from hedging_gym.config import TimeGrid
from hedging_gym.evaluation import empirical_es
from hedging_gym.finance import bank_subset, generate_market_bank
from methods.adaptation import train_multitask
from methods.meta_pretraining import (_assign_first_order_gradients,
    differentiable_empirical_es, split_source_bank, train_adapt_aware)
from methods.training import rollout


def test_fractional_empirical_es_scalar_and_gradient():
    losses = torch.tensor([5., 2., 10., 4., 9., 0., 3., 8., 1., 6.],
                          dtype=torch.float64, requires_grad=True)
    objective = differentiable_empirical_es(losses, .75)
    assert float(objective.detach()) == (10.+9.+.5*8.)/2.5
    assert float(objective.detach()) == empirical_es(losses, .75)
    objective.backward()
    torch.testing.assert_close(losses.grad, torch.tensor(
        [0., 0., .4, 0., .4, 0., 0., .2, 0., 0.], dtype=torch.float64))
    tiny = torch.tensor([1., 3., 2.], requires_grad=True)
    differentiable_empirical_es(tiny, .95).backward()
    torch.testing.assert_close(tiny.grad, torch.tensor([0., 1., 0.]))
    standard = torch.arange(1024, dtype=torch.float64, requires_grad=True)
    standard_es = differentiable_empirical_es(standard, .95)
    assert float(standard_es.detach()) == empirical_es(standard, .95)
    standard_es.backward()
    expected = torch.zeros_like(standard)
    expected[-51:] = 1/51.2
    expected[-52] = .2/51.2
    torch.testing.assert_close(standard.grad, expected, rtol=1e-12, atol=1e-15)


def test_first_order_mapping_is_post_update_query_gradient():
    # A scalar nonlinear support update makes the exact Jacobian nonidentity;
    # the expected reference below intentionally follows stop-gradient MAML.
    initial = nn.Module()
    initial.shared = nn.Linear(1, 1, bias=False).double()
    initial.source_embeddings = nn.Parameter(torch.tensor([[.2], [.7]], dtype=torch.float64))
    initial.embedding = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    with torch.no_grad():
        initial.shared.weight.fill_(.5)
    fast = deepcopy(initial)
    with torch.no_grad():
        fast.embedding.copy_(initial.source_embeddings[1])
    inner = torch.optim.Adam([*fast.shared.parameters(), fast.embedding], lr=.1)
    support_loss = (fast.shared.weight.square().sum()+fast.embedding.square().sum())
    support_loss.backward()
    inner.step()
    x = torch.tensor([[2.]], dtype=torch.float64)
    query_prediction = fast.shared(x).squeeze()+3*fast.embedding.squeeze()
    query_loss = .5*(query_prediction-1).square()
    residual = query_prediction.detach()-1
    _assign_first_order_gradients(initial, fast, 1, query_loss)
    torch.testing.assert_close(initial.shared.weight.grad, (2*residual).reshape(1, 1))
    torch.testing.assert_close(initial.source_embeddings.grad,
        torch.stack((residual.new_zeros(1), (3*residual).reshape(1))))
    # The gradient is neither a fast-weight displacement nor a support gradient.
    assert not torch.allclose(initial.shared.weight.grad,
                              initial.shared.weight.detach()-fast.shared.weight.detach())
    assert initial.shared.weight.grad.grad_fn is None
    assert initial.embedding.grad is None


def test_source_split_and_full_outer_optimizer_resume_are_exact(tmp_path):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2))
    other = replace(config, market=replace(config.market, v0=.0625))
    banks = (generate_market_bank(config, 24, 151, dtype=torch.float64),
             generate_market_bank(other, 24, 152, dtype=torch.float64))
    support, query = split_source_bank(banks[0])
    torch.testing.assert_close(support.spot, banks[0].spot[:12])
    torch.testing.assert_close(query.spot, banks[0].spot[12:])
    assert support.spot.storage_offset()+support.spot.numel() == query.spot.storage_offset()
    policy, original_metadata = train_multitask(banks, updates=2, batch_size=8,
        hidden=(4,), embedding_dim=2, seed=31, progress=False)
    original = deepcopy(policy.state_dict())
    options = dict(seed=41, inner_updates=2, batch_size=8, query_size=8, progress=False)
    full, full_metadata = train_adapt_aware(deepcopy(policy), original_metadata, banks,
        episodes=4, **options)
    checkpoint = tmp_path / "meta-latest.pt"
    train_adapt_aware(deepcopy(policy), original_metadata, banks,
        episodes=2, checkpoint_path=checkpoint, **options)
    resumed, metadata = train_adapt_aware(deepcopy(policy), original_metadata, banks,
        episodes=4, resume_from=checkpoint, checkpoint_path=checkpoint, **options)
    for name, value in full.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0., atol=0.)
        else:
            assert value == resumed.state_dict()[name]
    assert metadata["meta_pretraining"]["work"] == full_metadata["meta_pretraining"]["work"]
    assert metadata["source_zetas"] == full_metadata["source_zetas"]
    assert metadata["options"] == original_metadata["options"]
    assert metadata["initial_pretraining"] == original_metadata
    work = metadata["meta_pretraining"]["work"]
    assert work["source_unique_paths"] == 48
    assert work["initialization_forward_rollouts"] == 48
    assert work["support_gradient_rollouts"] == 64
    assert work["query_gradient_rollouts"] == 32
    assert work["total_episode_rollouts"] == 144
    assert not torch.equal(original["source_embeddings"], resumed.source_embeddings)
    torch.testing.assert_close(resumed.embedding, resumed.source_embeddings.mean(0))
    assert all(not parameter.requires_grad for parameter in resumed.shared.parameters())
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["step"] == 4
    assert all(float(state["step"]) == 4 for state in saved["optimizer"]["state"].values())

    # K0 must equal an ordinary gradient at the source initialization, including
    # drawing from all available source paths and using no support forwards.
    reference = deepcopy(policy)
    reference.shared.requires_grad_(True)
    reference.source_embeddings.requires_grad_(True)
    reference.embedding.requires_grad_(False)
    reference.active_task = 0
    parameters = [*reference.shared.parameters(), reference.source_embeddings]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    indices = torch.randperm(24, generator=torch.Generator().manual_seed(41+200003))[:8]
    loss = differentiable_empirical_es(
        rollout(reference, bank_subset(banks[0], indices))["terminal_loss"], config.risk.alpha)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    optimizer.step()
    reference.prepare_adaptation()
    control, control_metadata = train_adapt_aware(deepcopy(policy), original_metadata, banks,
        episodes=1, **{**options, "inner_updates": 0})
    for name, value in reference.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0., atol=0.)
        else:
            assert value == control.state_dict()[name]
    control_work = control_metadata["meta_pretraining"]["work"]
    assert control_work["initialization_forward_rollouts"] == 0
    assert control_work["support_gradient_rollouts"] == 0
    assert control_work["source_query_paths"] == [24, 24]


def test_ru_continuation_matches_direct_gradient_and_resumes_thresholds(tmp_path):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2))
    other = replace(config, market=replace(config.market, v0=.0625))
    banks = (generate_market_bank(config, 24, 151, dtype=torch.float64),
             generate_market_bank(other, 24, 152, dtype=torch.float64))
    policy, metadata = train_multitask(banks, updates=2, batch_size=8,
        hidden=(4,), embedding_dim=2, seed=31, progress=False)
    options = dict(seed=41, inner_updates=0, query_size=8, outer_loss="ru",
                   outer_lr=1e-4, zeta_lr=3e-4, progress=False)

    # A direct ordinary source-policy RU update must equal the gradient mapped
    # from the K0 clone. Both clip policy and threshold groups separately.
    reference = deepcopy(policy)
    reference.shared.requires_grad_(True)
    reference.source_embeddings.requires_grad_(True)
    reference.embedding.requires_grad_(False)
    reference.active_task = 0
    parameters = [*reference.shared.parameters(), reference.source_embeddings]
    thresholds = nn.Parameter(torch.tensor(metadata["source_zetas"], dtype=torch.float64))
    optimizer = torch.optim.Adam([
        {"params": parameters, "lr": options["outer_lr"]},
        {"params": [thresholds], "lr": options["zeta_lr"]},
    ])
    indices = torch.randperm(24, generator=torch.Generator().manual_seed(41+200003))[:8]
    losses = rollout(reference, bank_subset(banks[0], indices))["terminal_loss"]
    objective = config.risk.loss(losses, thresholds[0]).mean()
    objective.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_([thresholds], 5., error_if_nonfinite=True)
    optimizer.step()
    reference.prepare_adaptation()
    control, control_metadata = train_adapt_aware(deepcopy(policy), metadata, banks,
        episodes=1, **options)
    for name, value in reference.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0., atol=0.)
        else:
            assert value == control.state_dict()[name]
    assert control_metadata["source_zetas"] == thresholds.detach().tolist()
    record = control_metadata["meta_pretraining"]["history"][0]
    assert record["ru_objective"] == float(objective.detach())
    assert record["query_es"] == float(differentiable_empirical_es(losses.detach(), config.risk.alpha))

    full, full_metadata = train_adapt_aware(deepcopy(policy), metadata, banks,
        episodes=4, **options)
    checkpoint = tmp_path / "ru-latest.pt"
    train_adapt_aware(deepcopy(policy), metadata, banks,
        episodes=2, checkpoint_path=checkpoint, **options)
    resumed, resumed_metadata = train_adapt_aware(deepcopy(policy), metadata, banks,
        episodes=4, checkpoint_path=checkpoint, resume_from=checkpoint, **options)
    for name, value in full.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0., atol=0.)
        else:
            assert value == resumed.state_dict()[name]
    assert resumed_metadata["source_zetas"] == full_metadata["source_zetas"]
    assert resumed_metadata["meta_pretraining"]["work"] == full_metadata["meta_pretraining"]["work"]
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["thresholds"].tolist() == full_metadata["source_zetas"]
    assert len(saved["optimizer"]["param_groups"]) == 2
    assert all(float(state["step"]) == 4 for state in saved["optimizer"]["state"].values())
