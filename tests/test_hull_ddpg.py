import numpy as np
import torch

from hedging_gym.baselines.hull_ddpg import (
    HullLearner,
    KerasBatchNorm,
    PrioritizedReplay,
    SourceAdam,
    moment_targets,
    risk_score,
    train_hull_ddpg,
)
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import TimeGrid
from hedging_gym.environment.finance import generate_market_bank


def test_moment_targets_and_pointwise_risk():
    reward = torch.tensor([2., -3.])
    terminal = torch.tensor([False, True])
    first, second = moment_targets(reward, terminal, torch.tensor([5., 9.]), torch.tensor([30., 100.]))
    torch.testing.assert_close(first, torch.tensor([7., -3.]))
    torch.testing.assert_close(second, torch.tensor([54., 9.]))
    multiplier = 10000.
    scaled_first, scaled_second = moment_targets(multiplier*reward, terminal,
        multiplier*torch.tensor([5., 9.]), multiplier**2*torch.tensor([30., 100.]))
    torch.testing.assert_close(scaled_first, multiplier*first)
    torch.testing.assert_close(scaled_second, multiplier**2*second)
    # Appending another state must not change the first state's score or gradient.
    mean = torch.tensor([1., 2.], requires_grad=True)
    score = risk_score(mean, torch.tensor([5., 13.]))
    torch.testing.assert_close(score, torch.tensor([-2., -2.5]))
    gradient, = torch.autograd.grad(score.sum(), mean)
    torch.testing.assert_close(gradient, torch.tensor([1.75, 2.]))


def test_keras_bn_and_source_adam_equations():
    bn = KerasBatchNorm(2)
    data = torch.tensor([[1., 4.], [3., 8.], [5., 12.]])
    actual = bn(data)
    variance, mean = torch.var_mean(data, dim=0, correction=0)
    torch.testing.assert_close(actual, (data-mean)/torch.sqrt(variance+.001))
    torch.testing.assert_close(bn.running_mean, .01*mean)
    torch.testing.assert_close(bn.running_var, .99+.01*variance*3/(3-1.001))
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    optimizer = SourceAdam([parameter], .01, 1e-7)
    first, second, expected = 0., 0., 1.
    for step, gradient in enumerate((2., -1.), 1):
        first, second = .9*first+.1*gradient, .999*second+.001*gradient**2
        expected -= .01*np.sqrt(1-.999**step)/(1-.9**step)*first/(np.sqrt(second)+1e-7)
        optimizer.step((torch.tensor([gradient], dtype=torch.float64),))
        torch.testing.assert_close(parameter, torch.tensor([expected], dtype=torch.float64))


def test_priority_tree_covers_newest_slot_and_last_duplicate_wins():
    replay = PrioritizedReplay(5, alpha=1.)
    replay.add((torch.arange(5, dtype=torch.float64)[:, None],))
    replay.priorities(np.array([1, 1, 4]), np.array([7., 2., 20.]))
    assert replay.sums[replay.leaves+1] == 2.
    assert replay.sums[1] == 25.
    (values, weights), indices = replay.sample(128, .4, np.random.default_rng(2))
    assert 4 in indices and set(indices) <= set(range(5))
    torch.testing.assert_close(values.flatten(), torch.from_numpy(indices).double())
    torch.testing.assert_close(weights, torch.tensor((1/replay.sums[indices+replay.leaves])**.4))


def test_targets_copy_and_actor_bn_inference_mode():
    learner = HullLearner(benchmark_config())
    for source, target in zip(learner.actor.parameters(), learner.target_actor.parameters()):
        assert torch.equal(source, target)
    for source, target in zip(learner.critic.parameters(), learner.target_critic.parameters()):
        assert torch.equal(source, target)
    assert not learner.actor.training and not learner.actor.actor.input_bn.training
    assert learner.first_optimizer.parameters[0] is learner.second_optimizer.parameters[0]


def test_short_training_resume_preserves_update_ratio_and_state(tmp_path):
    torch.set_num_threads(1)
    config = benchmark_config(time_grid=TimeGrid(n_steps=2))
    bank = generate_market_bank(config, 72, 981)
    full, full_metadata = train_hull_ddpg(bank, episodes=72, collection_batch=8,
        replay_capacity=256, seed=19, progress=False, checkpoint_path=tmp_path/'full.pt')
    train_hull_ddpg(bank, episodes=64, collection_batch=8, replay_capacity=256,
        seed=19, progress=False, checkpoint_path=tmp_path/'split.pt')
    resumed, metadata = train_hull_ddpg(bank, episodes=72, collection_batch=8,
        replay_capacity=256, seed=19, progress=False, resume_from=tmp_path/'split.pt',
        checkpoint_path=tmp_path/'split.pt')
    assert metadata['learner_updates'] == full_metadata['learner_updates'] == 16
    assert metadata['critic_optimizer_steps'] == 128
    for key, value in full.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, resumed.state_dict()[key]), key
    a = torch.load(tmp_path/'full.pt', weights_only=False)
    b = torch.load(tmp_path/'split.pt', weights_only=False)
    for key, value in a['learner']['critic'].items():
        assert torch.equal(value, b['learner']['critic'][key]), key
