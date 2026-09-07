"""Adaptation must change the advertised parameters, never evaluation state."""

from dataclasses import replace

import numpy as np
import torch

from hedging_gym.benchmark import benchmark_config, evaluate_adaptation
from hedging_gym.config import TimeGrid
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import generate_market_bank
from methods.adaptation import AdaptationUpdater, train_multitask, train_online_finetune
from methods.controllers import policy_controller


def _banks():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    other = replace(config, market=replace(config.market, v0=.0625))
    return (generate_market_bank(config, 24, 101, dtype=torch.float64),
            generate_market_bank(other, 24, 102, dtype=torch.float64))


def test_online_finetuning_changes_weights_and_preserves_adam_across_stages():
    source, changed = _banks()
    policy, metadata = train_online_finetune(source, updates=2, batch_size=16,
        hidden=(8,), seed=17, progress=False)
    update = AdaptationUpdater(policy, metadata=metadata, updates=2, batch_size=16,
                               seed=19, progress=False)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    optimizer = update.optimizer
    update(source)
    update(changed)
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))
    assert update.optimizer is optimizer
    assert all(float(state["step"]) == 4 for state in optimizer.state.values())
    assert all(record["reset_embedding"] is False for record in update.history)


def test_embedding_updates_leave_shared_structure_frozen_and_evaluation_causal():
    banks = _banks()
    policy, metadata = train_multitask(banks, updates=4, batch_size=16,
        hidden=(8,), embedding_dim=2, seed=23, progress=False)
    shared_before = [parameter.detach().clone() for parameter in policy.shared.parameters()]
    embeddings_before = policy.source_embeddings.detach().clone()
    torch.testing.assert_close(policy.embedding, embeddings_before.mean(dim=0))
    assert metadata["updates_per_source"] == [2, 2]
    update = AdaptationUpdater(policy, metadata=metadata, updates=3, batch_size=16,
                               seed=29, progress=False)
    update(banks[0])
    assert not torch.equal(policy.embedding, embeddings_before.mean(dim=0))
    optimizer = update.optimizer
    update(banks[0])
    assert update.optimizer is optimizer  # Current-task continuation retains Adam.
    update(banks[1])
    assert update.optimizer is not optimizer  # Source protocol starts a new task.
    assert [record["reset_embedding"] for record in update.history] == [True, False, True]
    for original, parameter in zip(shared_before, policy.shared.parameters()):
        assert not parameter.requires_grad
        torch.testing.assert_close(parameter, original, rtol=0., atol=0.)
    torch.testing.assert_close(policy.source_embeddings, embeddings_before, rtol=0., atol=0.)
    before_evaluation = [parameter.detach().clone() for parameter in policy.parameters()]
    heldout = generate_market_bank(banks[1].config, 13, 203, dtype=torch.float64)
    metrics, tape = evaluate_controller(policy_controller(policy), heldout, batch_size=5)
    assert np.isfinite(metrics["expected_shortfall"])
    assert tape["positions"].shape[-1] == heldout.config.n_assets
    for original, parameter in zip(before_evaluation, policy.parameters()):
        torch.testing.assert_close(parameter, original, rtol=0., atol=0.)


def test_both_update_callbacks_run_chronological_a_b_a_on_common_heston():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    source = generate_market_bank(config, 16, 301)
    source_other = generate_market_bank(replace(config, market=replace(config.market,
        v0=.0625, theta=.0625)), 16, 302)
    direct, direct_metadata = train_online_finetune(source, updates=2, batch_size=8,
                                                  hidden=(8,), progress=False)
    embedded, embedded_metadata = train_multitask((source, source_other), updates=2,
                                                 batch_size=8, hidden=(8,), progress=False)
    for policy, metadata in ((direct, direct_metadata), (embedded, embedded_metadata)):
        update = AdaptationUpdater(policy, metadata=metadata, updates=1,
                                   batch_size=8, progress=False)
        report, tapes = evaluate_adaptation(policy_controller(policy), train_paths=8,
            eval_paths=8, seed=401, base_config=config, updates_per_stage=1,
            update=update, batch_size=8)
        assert [stage["stage"] for stage in report["stages"]] == ["A", "B", "A_return"]
        assert len(update.history) == 3
        assert all(stage["update_callback_calls"] == 1 for stage in report["stages"])
        for stage in report["stages"]:
            assert np.isfinite(stage["metrics"]["expected_shortfall"])
            assert stage["metrics"]["constraint_violations"] == 0
            assert tapes[stage["stage"]]["positions"].shape == (8, 3, config.n_assets)
