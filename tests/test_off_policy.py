"""Behavior at the replay/ledger boundary and exact CPU training continuation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("sb3_contrib")
from hedging_gym import benchmark_config
from hedging_gym.adapters.sb3 import SB3HedgingVecEnv
from hedging_gym.baselines._shared.sb3_off_policy import (
    TerminalRiskReplayBuffer,
    off_policy_controller,
    save_off_policy,
    set_risk_threshold,
)
from hedging_gym.baselines.crossq import build_crossq
from hedging_gym.baselines.crossq import load as load_crossq
from hedging_gym.baselines.tqc import build_tqc
from hedging_gym.baselines.tqc import load as load_tqc
from hedging_gym.environment.config import RiskConfig, TimeGrid
from hedging_gym.environment.finance import generate_market_bank, numpy_ledger
from hedging_gym.evaluation import evaluate_controller


def test_replay_relabels_only_terminal_rewards_at_current_threshold():
    bank = generate_market_bank(benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2)), 4, 42)
    env = SB3HedgingVecEnv(bank, 2, risk_threshold=.02)
    replay = TerminalRiskReplayBuffer(8, env.observation_space, env.action_space,
                                     n_envs=2, device="cpu", risk_threshold=.02)
    env.seed(9)
    observed = env.reset()
    for _ in range(2):
        action = np.zeros((2, bank.config.n_assets), np.float32)
        following, rewards, dones, infos = env.step(action)
        terminal = np.stack([info.get("terminal_observation", following[i]) for i,info in enumerate(infos)])
        replay.add(observed, terminal, action, rewards, dones, infos)
        observed = following
    np.testing.assert_allclose(replay.terminal_losses[1], [info["terminal_loss"] for info in infos])
    # Repeat the exact sample selection: only the objective threshold changes.
    batch_indices = np.array([0, 1]*20)
    np.random.seed(72)
    env_indices = np.random.randint(0, 2, size=len(batch_indices))
    np.random.seed(72)
    before = replay._get_samples(batch_indices)
    raw = replay.terminal_losses.copy()
    replay.risk_threshold = .12
    np.random.seed(72)
    after = replay._get_samples(batch_indices)
    assert torch.all(before.rewards[::2] == 0)
    assert torch.all(after.rewards[::2] == 0)
    assert not torch.equal(before.rewards[1::2], after.rewards[1::2])
    np.testing.assert_array_equal(replay.terminal_losses, raw)
    assert torch.all(after.dones[1::2] == 1)
    np.testing.assert_array_equal(after.next_observations.numpy(),
                                 replay.next_observations[batch_indices, env_indices])
    losses = torch.from_numpy(raw[batch_indices, env_indices]).reshape(-1, 1)
    torch.testing.assert_close(after.rewards,
        torch.where(after.dones.bool(), -bank.config.risk.loss(losses, .12), 0.))


def test_es_replay_adapter_rejects_mse():
    config = replace(benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2)),
                     risk=RiskConfig(objective="mse"))
    env = SB3HedgingVecEnv(generate_market_bank(config, 4, 42), 2)
    with pytest.raises(ValueError, match="terminal ES only"):
        build_tqc(env)


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        for a,b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("algorithm", ["crossq", "tqc"])
def test_stock_training_resume_and_independent_ledger(algorithm, tmp_path):
    torch.set_num_threads(1)
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 12, 440)
    env = SB3HedgingVecEnv(bank, 2)
    model = {'crossq': build_crossq, 'tqc': build_tqc}[algorithm](env, seed=7, buffer_size=64,
        learning_starts=0, batch_size=4, policy_kwargs=dict(net_arch=[8,8]))
    model.learn(12)
    assert model._n_updates == 12  # One update per transition, not per vector step.
    set_risk_threshold(model, env, .03)
    save_off_policy(model, env, tmp_path / "checkpoint", runner_state={"phase": 1})
    model.learn(12, reset_num_timesteps=False)
    expected = deepcopy(model.get_parameters())
    expected_entropy = model.log_ent_coef.detach().clone()
    expected_replay = deepcopy(model.replay_buffer)
    expected_obs = model._last_obs.copy()
    loaded_env = SB3HedgingVecEnv(bank, 2)
    loaded, runner = {'crossq': load_crossq, 'tqc': load_tqc}[algorithm](tmp_path / "checkpoint", loaded_env)
    assert runner == {"phase": 1}
    assert loaded.replay_buffer.risk_threshold == loaded_env.risk_threshold == .03
    loaded.learn(12, reset_num_timesteps=False)
    assert loaded.num_timesteps == model.num_timesteps == 24
    assert loaded._n_updates == model._n_updates == 24
    assert_state_equal(expected, loaded.get_parameters())
    torch.testing.assert_close(expected_entropy, loaded.log_ent_coef, atol=0, rtol=0)
    np.testing.assert_array_equal(expected_obs, loaded._last_obs)
    for field in ("observations", "next_observations", "actions", "rewards", "dones"):
        np.testing.assert_array_equal(getattr(expected_replay, field), getattr(loaded.replay_buffer, field))
    for deterministic in (True, False):
        observed = torch.from_numpy(loaded._last_obs)
        # Identical sampling RNG isolates inference/action-coordinate parity.
        with torch.random.fork_rng():
            torch.manual_seed(823)
            prediction, _ = loaded.predict(loaded._last_obs, deterministic=deterministic)
            expected_targets = loaded_env.lower + (prediction+1.) * (loaded_env.upper-loaded_env.lower) / 2.
            torch.manual_seed(823)
            actual_targets = off_policy_controller(loaded, deterministic=deterministic)(
                observed, loaded_env.tensor_env.state, 0, config)
        np.testing.assert_array_equal(actual_targets.numpy(), expected_targets)
        _, tape = evaluate_controller(off_policy_controller(loaded, deterministic=deterministic), bank)
        reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
            bank.liability[:,0].numpy(), bank.liability[:,-1].numpy(), config)
        np.testing.assert_allclose(reference["terminal_loss"], tape["terminal_loss"], atol=2e-6, rtol=0)
    loaded_env.step(np.zeros((2,config.n_assets), np.float32))
    with pytest.raises(ValueError, match="completed episode batch"):
        save_off_policy(loaded, loaded_env, tmp_path / "partial")
