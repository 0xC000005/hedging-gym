"""Retrieval learns source transfer, not target evaluation results."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from hedging_gym.baselines._shared.training import rollout
from hedging_gym.baselines.adaptive_deep_hedging import TaskEmbeddedPolicy
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import TimeGrid
from hedging_gym.environment.finance import generate_market_bank
from hedging_gym.evaluation import empirical_es
from hedging_gym.extensions.skill_retrieval import (
    TransferRiskPredictor,
    nearest_context,
    rank_contexts,
    score_source_contexts,
    select_context,
    train_retriever,
)


def _example():
    base = benchmark_config(time_grid=TimeGrid(n_steps=3))
    changed = replace(base, market=replace(base.market, v0=.09, theta=.09))
    banks = [generate_market_bank(config, 32, seed, dtype=torch.float64)
             for config, seed in zip((base, changed), (7301, 7302))]
    torch.manual_seed(71)
    policy = TaskEmbeddedPolicy(base, n_tasks=2, embedding_dim=2, hidden=(8,)).double()
    with torch.no_grad():
        policy.source_embeddings.copy_(torch.tensor([[-1., .5], [1., -.5]]))
    policy.prepare_adaptation()
    return policy, banks


def test_source_scores_pool_paths_and_leave_source_policy_unchanged():
    policy, banks = _example()
    before = deepcopy(policy.state_dict())
    flags = [parameter.requires_grad for parameter in policy.parameters()]
    scores, mean_scores, work = score_source_contexts(policy, banks, chunk_size=7, progress=False)
    independent = deepcopy(policy)
    with torch.no_grad():
        for target, bank in enumerate(banks):
            for source in range(2):
                independent.embedding.copy_(independent.source_embeddings[source])
                actual = empirical_es(rollout(independent, bank)["terminal_loss"], bank.config.risk.alpha)
                torch.testing.assert_close(scores[source, target], torch.tensor(actual).double(),
                                           rtol=1e-6, atol=1e-10)
            independent.reset_embedding()
            actual = empirical_es(rollout(independent, bank)["terminal_loss"], bank.config.risk.alpha)
            torch.testing.assert_close(mean_scores[target], torch.tensor(actual).double(),
                                       rtol=1e-6, atol=1e-10)
    assert work["episode_rollouts"] == 3*64
    assert work["ledger_decisions"] == 3*3*64
    assert flags == [parameter.requires_grad for parameter in policy.parameters()]
    assert policy.active_task is None
    for name, value in policy.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before[name], rtol=0., atol=0.)
        else:
            assert value == before[name]


def test_pairwise_head_has_native_relu_mse_gradient_and_directional_inputs():
    policy, banks = _example()
    retriever = TransferRiskPredictor([bank.config for bank in banks], hidden=3).double()
    source = retriever.source_features[[0, 1]].clone().requires_grad_()
    target = retriever.source_features[[1, 0]].clone().requires_grad_()
    labels = torch.tensor([.2, -.3], dtype=torch.float64)
    prediction = retriever(source, target)
    inputs = torch.cat(((source-retriever.feature_mean)/retriever.feature_scale,
                        (target-retriever.feature_mean)/retriever.feature_scale), -1)
    hidden = (inputs@retriever.head[0].weight.T+retriever.head[0].bias).relu()
    expected = (hidden@retriever.head[2].weight.T+retriever.head[2].bias).squeeze(-1)
    actual_gradient = torch.autograd.grad((prediction-labels).square().mean(),
                                          (source, target), retain_graph=True)
    expected_gradient = torch.autograd.grad((expected-labels).square().mean(), (source, target))
    torch.testing.assert_close(prediction, expected, rtol=0., atol=0.)
    for actual, wanted in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, wanted, rtol=0., atol=0.)


def test_retrieval_ranks_predicted_risk_without_mutating_context():
    policy, banks = _example()
    retriever = TransferRiskPredictor([bank.config for bank in banks], hidden=1).double()
    index = retriever.fields.index("v0")
    with torch.no_grad():
        retriever.head[0].weight.zero_()
        retriever.head[0].weight[0, index] = 1.
        retriever.head[0].bias.fill_(2.)
        retriever.head[2].weight.fill_(-1.)
        retriever.head[2].bias.zero_()
    before = policy.embedding.detach().clone()
    assert rank_contexts(retriever, policy, banks[0].config) == [1, 0]
    assert select_context(retriever, policy, banks[0].config) == 1
    assert nearest_context(retriever, banks[0].config) == 0
    assert nearest_context(retriever, banks[1].config) == 1
    torch.testing.assert_close(policy.embedding, before, rtol=0., atol=0.)


def test_retrieval_uses_numeric_market_features_and_rejects_scheme_changes():
    base = benchmark_config()
    other = replace(base, market=replace(base.market, v0=.09))
    retriever = TransferRiskPredictor((base, other))
    assert retriever.source_features.is_floating_point()
    assert retriever.features(other)[retriever.fields.index("v0")].item() == pytest.approx(.09)
    assert torch.isfinite(retriever.predict_scores(other)).all()
    changed = replace(other, market=replace(other.market, scheme="qe"))
    with pytest.raises(ValueError, match="source tasks.*scheme"):
        TransferRiskPredictor((base, changed))
    with pytest.raises(ValueError, match="target.*scheme"):
        retriever.predict_scores(changed)


def test_fit_preserves_labels_optimizer_and_source_only_normalization(tmp_path):
    policy, banks = _example()
    path = tmp_path/"retriever.pt"
    retriever, metadata = train_retriever(policy, banks, updates=100, hidden=8,
        seed=79, progress=False, checkpoint_path=path)
    assert metadata["history"][-1]["normalized_mse"] < metadata["history"][0]["normalized_mse"]
    scores = torch.tensor(metadata["transfer_es"], dtype=torch.float64)
    means = torch.tensor(metadata["mean_context_es"], dtype=torch.float64)
    torch.testing.assert_close(torch.tensor(metadata["relative_es"], dtype=torch.float64),
                               scores-means.unsqueeze(0))
    saved = torch.load(path, weights_only=False)
    assert saved["step"] == 100
    assert all(int(value["step"]) == 100 for value in saved["optimizer"]["state"].values())
    restored = TransferRiskPredictor([bank.config for bank in banks], hidden=8).double()
    restored.load_state_dict(saved["predictor"])
    new_market = replace(banks[0].config, market=replace(banks[0].config.market, v0=.0625))
    old_mean = restored.feature_mean.clone()
    torch.testing.assert_close(restored.predict_scores(new_market), retriever.predict_scores(new_market))
    torch.testing.assert_close(restored.feature_mean, old_mean, rtol=0., atol=0.)
