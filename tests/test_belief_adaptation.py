"""Qualify the donor encoder and its causal, terminal-risk DH integration."""

from copy import deepcopy
from dataclasses import replace
from functools import partial

import pytest
import torch

from hedging_gym.benchmark import benchmark_config
from hedging_gym.config import TimeGrid
from hedging_gym.finance import generate_market_bank, observation_fields
from methods.adaptation import AdaptationUpdater
from methods.belief_adaptation import (BeliefDynamicsEncoder, BeliefEmbeddedPolicy,
    encode_bank_context, train_belief_encoder)
from methods.training import rollout


def _banks(seed=601):
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    other = replace(config, market=replace(config.market, v0=.09, theta=.09,
                                           sigma=.7, rho=-.35))
    return tuple(generate_market_bank(current, 12, seed+i, dtype=torch.float64)
                 for i, current in enumerate((config, other)))


def _encoder(banks):
    return train_belief_encoder(banks, updates=4, batch_size=4, context_length=2,
        width=12, heads=2, layers=1, mlp_dim=16, embedding_dim=2, seed=607, progress=False)


def test_native_transition_set_architecture_is_permutation_invariant_and_learns():
    torch.manual_seed(593)
    encoder = BeliefDynamicsEncoder(embedding_dim=2, width=12, heads=2,
                                    layers=1, mlp_dim=16).double()
    states, actions, next_states = torch.randn(4, 5, 2).double(), torch.zeros(4, 5, 1).double(), torch.randn(4, 5, 2).double()
    original = encoder(states, actions, next_states)
    order = torch.tensor([3, 0, 4, 2, 1])
    permuted = encoder(states[:, order], actions[:, order], next_states[:, order])
    for first, second in zip(original, permuted):
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
    encoder.prediction_loss(states, actions, next_states).backward()
    assert encoder.state_projection.weight.grad.abs().sum() > 0
    assert encoder.context_mean.weight.grad.abs().sum() > 0
    assert encoder.context_log_std.weight.grad.abs().sum() > 0
    assert encoder.predictor[-1].weight.grad.abs().sum() > 0


def test_context_training_is_reproducible_and_inference_uses_only_declared_prior_prefix():
    banks = _banks()
    encoder, metadata = _encoder(banks)
    again, _ = _encoder(banks)
    for first, second in zip(encoder.parameters(), again.parameters()):
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
        assert not first.requires_grad
    assert metadata["transition_presentations"] == 4 * 4 * 2
    vector, cost = encode_bank_context(encoder, banks[0], history_paths=3, context_length=2)
    changed = deepcopy(banks[0])
    changed.spot[3:] *= 10  # Unselected prior paths.
    changed.variance[3:] *= 10
    changed.spot[:3, 3:] *= 10  # Dates after the declared prior context.
    changed.variance[:3, 3:] *= 10
    changed.liability.fill_(999.)
    changed.marks.fill_(999.)  # Encoder has no financial-label input.
    other, _ = encode_bank_context(encoder, changed, history_paths=3, context_length=2)
    torch.testing.assert_close(vector, other, rtol=0., atol=0.)
    assert cost["history_transitions"] == 6
    with pytest.raises(ValueError, match="insufficient"):
        encode_bank_context(encoder, banks[0], history_paths=13, context_length=2)


def test_complete_book_es_gradients_adapt_only_context_and_keep_observed_parameters(tmp_path):
    histories = _banks()
    encoder, _ = _encoder(histories)
    contexts = torch.stack([encode_bank_context(encoder, bank, history_paths=3,
                                               context_length=2)[0] for bank in histories])
    factory = partial(BeliefEmbeddedPolicy, encoder=encoder, source_contexts=contexts)
    policy = factory(histories[0].config, n_tasks=2, embedding_dim=2, hidden=(8,)).double()
    assert policy.observation_fields == tuple(observation_fields(histories[0].config))
    assert {"kappa", "theta", "sigma", "rho", "v0"} <= set(policy.observation_fields)
    source_before = policy.source_embeddings.detach().clone()
    encoder_before = deepcopy(policy.encoder.state_dict())
    # Shared pretraining uses the same complete-book terminal ES objective and
    # the same active_task contract as train_multitask's common factory.
    train_banks = _banks(seed=619)
    shared_before = deepcopy(policy.shared.state_dict())
    zeta = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([*policy.shared.parameters(), policy.source_embeddings, zeta], lr=.001)
    for task, bank in enumerate(train_banks):
        policy.active_task = task
        terminal = rollout(policy, bank)["terminal_loss"]
        objective = bank.config.risk.loss(terminal, zeta[task]).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    assert any(not torch.equal(value, policy.shared.state_dict()[name]) for name, value in shared_before.items())
    torch.testing.assert_close(policy.source_embeddings, source_before, rtol=0., atol=0.)
    policy.prepare_adaptation()
    target_prior = _banks(seed=631)[1]
    policy.infer_context(target_prior, history_paths=3, context_length=2)
    inferred = policy.embedding.detach().clone()
    frozen_shared = deepcopy(policy.shared.state_dict())
    update = AdaptationUpdater(policy, updates=2, batch_size=8, seed=641, progress=False)
    torch.testing.assert_close(policy.embedding, inferred, rtol=0., atol=0.)
    update(train_banks[1])
    assert not torch.equal(policy.embedding, inferred)
    for name, original in frozen_shared.items():
        torch.testing.assert_close(policy.shared.state_dict()[name], original, rtol=0., atol=0.)
    for name, original in encoder_before.items():
        torch.testing.assert_close(policy.encoder.state_dict()[name], original, rtol=0., atol=0.)
    torch.testing.assert_close(policy.prior_context, inferred, rtol=0., atol=0.)
    heldout = _banks(seed=653)[1]
    before_eval = deepcopy(policy.state_dict())
    with torch.no_grad():
        result = rollout(policy, heldout, record_positions=True)
    assert result["positions"].shape == (12, 3, heldout.config.n_assets)
    assert torch.isfinite(result["terminal_loss"]).all()
    for name, value in policy.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before_eval[name], rtol=0., atol=0.)
    checkpoint = tmp_path / "belief-policy.pt"
    torch.save(policy.state_dict(), checkpoint)
    restored = factory(histories[0].config, n_tasks=2, embedding_dim=2, hidden=(8,)).double()
    restored.load_state_dict(torch.load(checkpoint, weights_only=False))
    restored.prepare_adaptation()
    torch.testing.assert_close(restored.embedding, inferred, rtol=0., atol=0.)
    assert bool(restored.has_target_history)
