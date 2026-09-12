"""Check objective preservation and interruption safety, not performance claims."""
import pytest
import torch

from hedging_gym import benchmark_config
from hedging_gym.config import RiskConfig, TimeGrid
from hedging_gym.benchmark import adaptation_configs
from hedging_gym.finance import generate_market_bank
from methods.curriculum import pooled_ru_comparison, task_probabilities, train_task_curriculum
from methods.policies import NoTransactionBandPolicy


def test_task_weight_preserves_pooled_risk_and_threshold_gradient():
    zeta = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    losses = torch.tensor([[.2, .4, .8], [.1, 1.2, 2.]], dtype=torch.float64,
                          requires_grad=True)
    q = task_probabilities([1., 4.])
    objective = RiskConfig(.8).loss(losses, zeta).mean(1)
    expected_sampled = (q * (1 / (2*q)) * objective).sum()
    direct = RiskConfig(.8).loss(losses.flatten(), zeta).mean()
    torch.testing.assert_close(expected_sampled, direct)
    actual = torch.autograd.grad(expected_sampled, (losses, zeta), retain_graph=True)
    reference = torch.autograd.grad(direct, (losses, zeta))
    for value, target in zip(actual, reference):
        torch.testing.assert_close(value, target)
    torch.testing.assert_close(task_probabilities([0., -1.]), torch.tensor([.5, .5], dtype=torch.float64))


def test_teacher_is_chosen_after_expected_risk_not_per_path():
    # Each teacher loses on one path. A clairvoyant per-path teacher switch would
    # appear riskless, but neither complete feasible policy is better here.
    student = torch.tensor([[2., 2.]])
    teachers = torch.tensor([[[0., 4.], [4., 0.]]])
    current, reference, _ = pooled_ru_comparison(student, teachers, 0., RiskConfig(.5))
    torch.testing.assert_close(current, reference)
    assert reference.item() == 4.


@pytest.mark.parametrize("sampler", ["regret", "stratified", "pooled_ru_regret"])
def test_curriculum_resume_matches_uninterrupted(tmp_path, sampler):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    tasks = adaptation_configs(config)
    banks = tuple(generate_market_bank(c, 24, 441+i, dtype=torch.float64)
                  for i, (_, c) in enumerate(tasks[:2]))
    policy = NoTransactionBandPolicy(config, hidden=(8,)).double()
    options = dict(sampler=sampler, teacher_scores=[.01, .01], seed=7,
        batch_size=8, score_paths=12, score_every=2, progress=False)
    if sampler == "pooled_ru_regret":
        options["teacher_losses"] = torch.linspace(.01, .5, 48).reshape(2, 2, 12).double()
    full, fm = train_task_curriculum(policy, banks, updates=4, **options)
    path = tmp_path / "latest.pt"
    train_task_curriculum(policy, banks, updates=2, checkpoint_path=path, **options)
    legacy = torch.load(path, weights_only=False)
    for config in legacy["source_configs"]:
        config["time_grid"].pop("trade_at_maturity")
        config["time_grid"].pop("step_days")
        config.pop("settlement")
    torch.save(legacy, path)
    resumed, rm = train_task_curriculum(policy, banks, updates=4,
        checkpoint_path=path, resume_from=path, **options)
    for name, value in full.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        else:
            assert value == resumed.state_dict()[name]
    assert rm["zeta"] == fm["zeta"]
    assert rm["task_counts"] == fm["task_counts"]
    assert rm["schedule"] == fm["schedule"]
