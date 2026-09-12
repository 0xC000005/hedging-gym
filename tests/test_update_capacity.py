"""Matched-checkpoint adaptation changes capacity without changing its start."""

from copy import deepcopy
from dataclasses import replace

import torch

from experiments.compare_update_capacity import matched_updater, path_usage
from hedging_gym.benchmark import benchmark_config
from hedging_gym.config import TimeGrid
from hedging_gym.finance import bank_subset, generate_market_bank
from methods import adaptation
from methods.adaptation import train_multitask
from methods.training import rollout


def _prepared():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    target = replace(config, market=replace(config.market, v0=.0625))
    banks = (generate_market_bank(config, 24, 101, dtype=torch.float64),
             generate_market_bank(target, 24, 102, dtype=torch.float64))
    policy, metadata = train_multitask(banks, updates=4, batch_size=16,
        hidden=(8,), embedding_dim=2, seed=23, progress=False)
    initial = policy.source_embeddings[1].detach().clone()
    scorer = deepcopy(policy)
    with torch.no_grad():
        scorer.embedding.copy_(initial)
        losses = rollout(scorer, bank_subset(banks[1], slice(0, 8)))["terminal_loss"]
        zeta = torch.quantile(losses, target.risk.alpha)
    return policy, metadata, banks[1], initial, zeta


def _assert_policy_equal(first, second):
    for name, parameter in first.state_dict().items():
        if torch.is_tensor(parameter):
            torch.testing.assert_close(parameter, second.state_dict()[name], rtol=0., atol=0.)
        else:
            assert parameter == second.state_dict()[name]


def test_matched_start_stream_and_distinct_update_capacity(monkeypatch):
    policy, metadata, train, initial, zeta = _prepared()
    original = deepcopy(policy)
    arms = {mode: matched_updater(policy, metadata, train, initial, zeta,
        mode=mode, seed=53, batch_size=8, updates=2)
        for mode in ("embedding", "finetune")}
    embedding, finetune = arms.values()
    _assert_policy_equal(embedding.policy, finetune.policy)
    for updater in arms.values():
        torch.testing.assert_close(updater.policy.embedding, initial, rtol=0., atol=0.)
        torch.testing.assert_close(updater.zeta, zeta, rtol=0., atol=0.)
        assert updater.last_market == train.config.market
        assert updater.policy.active_task is None
        assert not updater.optimizer.state
    with torch.no_grad():
        torch.testing.assert_close(rollout(embedding.policy, train)["terminal_loss"],
            rollout(finetune.policy, train)["terminal_loss"], rtol=0., atol=0.)

    sampled = []
    original_subset = adaptation.bank_subset

    def capture_subset(bank, indices):
        sampled.append(indices.detach().cpu().clone())
        return original_subset(bank, indices)

    monkeypatch.setattr(adaptation, "bank_subset", capture_subset)
    for updater in arms.values():
        updater(train)
        assert updater.history[0]["initialization_paths"] == 0
        assert updater.history[0]["reset_embedding"] is False
    assert len(sampled) == 4  # Two actual minibatches in each arm.
    for left, right in zip(sampled[:2], sampled[2:]):
        assert torch.equal(left, right)
    assert torch.equal(embedding.index_generator.get_state(),
                       finetune.index_generator.get_state())
    observed_paths = set(range(8))
    observed_paths.update(torch.cat(sampled[:2]).tolist())
    assert path_usage(53, len(train.spot), 8, 0, 8) == 8
    assert path_usage(53, len(train.spot), 8, 2, 8) == len(observed_paths)
    _assert_policy_equal(policy, original)
    for name, parameter in embedding.policy.shared.named_parameters():
        assert not parameter.requires_grad
        torch.testing.assert_close(parameter, original.shared.state_dict()[name],
                                   rtol=0., atol=0.)
    assert any(not torch.equal(parameter, original.shared.state_dict()[name])
               for name, parameter in finetune.policy.shared.named_parameters())
    for updater in arms.values():
        torch.testing.assert_close(updater.policy.source_embeddings,
                                   original.source_embeddings, rtol=0., atol=0.)


def test_matched_capacity_checkpoint_resume_preserves_training(tmp_path):
    policy, metadata, train, initial, zeta = _prepared()
    for mode in ("embedding", "finetune"):
        options = dict(mode=mode, seed=59, batch_size=8, updates=2)
        uninterrupted = matched_updater(policy, metadata, train, initial, zeta, **options)
        uninterrupted(train)
        checkpoint = tmp_path / f"{mode}.pt"
        torch.save(uninterrupted.state_dict(), checkpoint)
        uninterrupted(train)
        resumed = matched_updater(policy, metadata, train, initial, zeta, **options)
        resumed.load_state_dict(torch.load(checkpoint, weights_only=False))
        resumed(train)
        _assert_policy_equal(uninterrupted.policy, resumed.policy)
        torch.testing.assert_close(uninterrupted.zeta, resumed.zeta, rtol=0., atol=0.)
        assert resumed.completed_steps == uninterrupted.completed_steps == 4
        assert torch.equal(uninterrupted.index_generator.get_state(),
                           resumed.index_generator.get_state())
        assert [p.requires_grad for p in uninterrupted.policy.parameters()] == [
            p.requires_grad for p in resumed.policy.parameters()]
        assert all(record["initialization_paths"] == 0 for record in resumed.history)
        assert all(not record["reset_embedding"] for record in resumed.history)
