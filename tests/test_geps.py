"""Equation, pinned author-layer, and common-ledger adapter qualification."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from hedging_gym.benchmark import benchmark_config
from hedging_gym.config import TimeGrid
from hedging_gym.finance import generate_market_bank
from methods.adaptation import AdaptationUpdater, TaskEmbeddedPolicy
from methods.geps import GEPSLinear, GEPSPolicy
from methods.training import rollout


def test_geps_layer_matches_explicit_equations_and_gradients():
    torch.manual_seed(701)
    layer = GEPSLinear(5, 3, 2).double()
    inputs = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    context = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    efficient = layer(inputs, context)
    weights = layer.weight + layer.A @ torch.diag_embed(context) @ layer.B
    literal = (inputs.unsqueeze(1) @ weights).squeeze(1)
    literal = literal + layer.bias + context @ layer.bias_context
    torch.testing.assert_close(efficient, literal, rtol=1e-12, atol=1e-12)
    targets = (inputs, context, *layer.parameters())
    probe = torch.randn_like(efficient)
    efficient_grads = torch.autograd.grad((efficient * probe).sum(), targets)
    literal_grads = torch.autograd.grad((literal * probe).sum(), targets)
    for actual, expected in zip(efficient_grads, literal_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_geps_layer_matches_pinned_author_source():
    """Set GEPS_DONOR_ROOT to the retained author checkout; no donor install."""
    donor_root = os.environ.get("GEPS_DONOR_ROOT")
    if donor_root is None:
        pytest.skip("pinned author checkout not supplied via GEPS_DONOR_ROOT")
    path = Path(donor_root) / "geps/model/layers.py"
    spec = importlib.util.spec_from_file_location("geps_author_layers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(709)
    ours = GEPSLinear(5, 3, 2).double()
    source = module.GEPSLinear(5, 3, code=2, factor=1, dtype=torch.float64)
    source.load_state_dict(ours.state_dict())
    inputs = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    context = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    actual = ours(inputs, context)
    expected = source(inputs, torch.diag_embed(context))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(),
                                       (inputs, context, *ours.parameters()))
    expected_grads = torch.autograd.grad((expected * probe).sum(),
                                         (inputs, context, *source.parameters()))
    for first, second in zip(actual_grads, expected_grads):
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_geps_source_context_is_shared_by_all_layers_and_jointly_differentiable():
    torch.manual_seed(719)
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    policy = GEPSPolicy(config, n_tasks=3, embedding_dim=2, hidden=(8, 6)).double()
    assert torch.count_nonzero(policy.source_embeddings) == 0
    with torch.no_grad():
        policy.source_embeddings[1].copy_(torch.tensor([.2, -.1], dtype=torch.float64))
    policy.active_task = 1
    features = torch.randn(5, policy.feature_dim, dtype=torch.float64)
    holdings = torch.zeros(5, policy.n_assets, dtype=torch.float64)
    contexts = []
    handles = [layer.register_forward_pre_hook(lambda _layer, args: contexts.append(args[1]))
               for layer in policy.shared.layers]
    action = policy(features, holdings, -1., 1.)
    for handle in handles:
        handle.remove()
    assert len(contexts) == 3
    for context in contexts:
        assert context is contexts[0]
        torch.testing.assert_close(context, policy.source_embeddings[1].expand(5, -1))
    action.target_holdings.square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in policy.shared.parameters())
    assert policy.source_embeddings.grad[1].abs().sum() > 0
    assert torch.count_nonzero(policy.source_embeddings.grad[[0, 2]]) == 0
    assert policy.embedding.grad is None


def test_geps_common_es_update_fits_only_new_context_and_threshold():
    torch.manual_seed(727)
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    policy = GEPSPolicy(config, n_tasks=2, embedding_dim=4, hidden=(8,)).double()
    assert isinstance(policy, TaskEmbeddedPolicy)
    policy.prepare_adaptation()
    original = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    assert [name for name, parameter in policy.named_parameters() if parameter.requires_grad] == ["embedding"]
    torch.testing.assert_close(policy.embedding, policy.source_embeddings.mean(dim=0))
    bank = generate_market_bank(config, 16, 733, dtype=torch.float64)
    updater = AdaptationUpdater(policy, updates=2, batch_size=8, seed=739, progress=False)
    assert updater.mode == "embedding"
    assert updater.trainable == [policy.embedding]
    updater(bank)
    assert not torch.equal(policy.embedding, original["embedding"])
    for name, parameter in policy.named_parameters():
        if name != "embedding":
            assert not parameter.requires_grad
            torch.testing.assert_close(parameter, original[name], rtol=0., atol=0.)
    before_evaluation = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    with torch.no_grad():
        losses = rollout(policy, bank)["terminal_loss"]
    assert torch.isfinite(losses).all()
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before_evaluation[name], rtol=0., atol=0.)
