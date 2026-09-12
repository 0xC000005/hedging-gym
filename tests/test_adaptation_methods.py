"""Adaptation must change the advertised parameters, never evaluation state."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym.baselines._shared.adaptation import AdaptationUpdater
from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines.adaptive_deep_hedging import (
    TaskEmbeddedPolicy,
    train_multitask,
)
from hedging_gym.baselines.finetune_dh import train_online_finetune
from hedging_gym.environment.benchmark import benchmark_config, evaluate_adaptation
from hedging_gym.environment.config import TimeGrid
from hedging_gym.environment.finance import generate_market_bank
from hedging_gym.evaluation import evaluate_controller


def _banks():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    other = replace(config, market=replace(config.market, v0=.0625))
    return (generate_market_bank(config, 24, 101, dtype=torch.float64),
            generate_market_bank(other, 24, 102, dtype=torch.float64))


def test_embedding_adapter_rejects_unbounded_task_before_nan_actions():
    config = benchmark_config(model="gbm")
    config = replace(config, execution=replace(config.execution, holding_upper=None))
    with pytest.raises(ValueError, match="finite holding bounds"):
        TaskEmbeddedPolicy(config, n_tasks=2)


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


def test_multitask_and_mid_call_adaptation_resume_exactly(tmp_path):
    """A saved update resumes, not restarts, including frozen task structure."""
    banks = _banks()
    options = dict(batch_size=16, hidden=(8,), embedding_dim=2, seed=43, progress=False)
    full, full_metadata = train_multitask(banks, updates=6, **options)
    path = tmp_path / "pretrain.pt"
    train_multitask(banks, updates=3, checkpoint_path=path, **options)
    resumed, metadata = train_multitask(banks, updates=6, resume_from=path, **options)
    for first, second in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
        assert first.requires_grad == second.requires_grad
    assert metadata["source_zetas"] == full_metadata["source_zetas"]
    assert path.with_name("pretrain-early.pt").exists()

    adaptation_path = tmp_path / "adapt.pt"
    updater = AdaptationUpdater(full, metadata=full_metadata, updates=3, batch_size=16,
        seed=47, progress=False, checkpoint_path=adaptation_path)
    updater(banks[1])
    early = torch.load(tmp_path / "adapt-early.pt", weights_only=False)
    other = AdaptationUpdater(resumed, metadata=metadata, updates=3, batch_size=16,
                              seed=47, progress=False)
    other.load_state_dict(early["state"])
    assert other.pending_call["completed"] == 1
    other(banks[1])
    for first, second in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
        assert first.requires_grad == second.requires_grad
    torch.testing.assert_close(updater.zeta, other.zeta, rtol=0., atol=0.)
    assert other.completed_steps == 3
    assert len(other.history) == 1 and other.pending_call is None
    assert all(float(state["step"]) == 3 for state in other.optimizer.state.values())

    # Returning A is a fresh embedding calibration, not restoration of A's old vector.
    updater(banks[0])
    other(banks[0])
    for first, second in zip(full.parameters(), resumed.parameters()):
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
