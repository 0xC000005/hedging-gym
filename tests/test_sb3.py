"""SB3 boundary: complete financial episodes and checkpoint action parity."""
import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from hedging_gym import benchmark_config
from hedging_gym.config import TimeGrid
from hedging_gym.finance import generate_market_bank
from hedging_gym.gym_env import TensorHedgingEnv
from methods.sb3 import SB3HedgingVecEnv, sb3_controller


def test_sb3_ledger_reward_and_autoreset():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 12, 440)
    env = SB3HedgingVecEnv(bank, 4, risk_threshold=.02)
    env.seed(42)
    first = env.reset()
    reference = TensorHedgingEnv(env.tensor_env._bank)
    action = np.zeros((4, config.n_assets), np.float32)
    targets = torch.from_numpy(env.lower+(action+1)*(env.upper-env.lower)/2)
    for step in range(config.n_steps):
        observed, rewards, dones, infos = env.step(action)
        expected, _, done, _, result = reference.step(targets)
        assert dones.all() == done
        if not done:
            np.testing.assert_array_equal(observed, expected.numpy())
            assert not rewards.any()
    np.testing.assert_allclose(rewards, -config.risk.loss(result["terminal_loss"], .02))
    np.testing.assert_array_equal(infos[0]["terminal_observation"], expected[0].numpy())
    assert infos[0]["TimeLimit.truncated"] is False
    assert env.tensor_env.time_index == 0
    env.seed(42)
    np.testing.assert_array_equal(env.reset(), first)
    env.close()


def test_stock_ppo_checkpoint_actions(tmp_path):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 12, 440)
    env = SB3HedgingVecEnv(bank, 4, risk_threshold=.02)
    model = PPO("MlpPolicy", env, seed=7, n_steps=3, batch_size=12, n_epochs=1, device="cpu",
                gamma=1., gae_lambda=1., policy_kwargs=dict(net_arch=[8]))
    model.learn(24)
    obs = env.reset()
    actions = model.predict(obs, deterministic=True)[0]
    model.save(tmp_path / "ppo")
    loaded = PPO.load(tmp_path / "ppo", device="cpu")
    np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], actions)
    env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_ppo_actions_match_original_predict(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 12, 440)
    env = SB3HedgingVecEnv(bank, 4)
    model = PPO("MlpPolicy", env, seed=7, n_steps=3, batch_size=12, device=device,
                policy_kwargs=dict(net_arch=[8]))
    batch = torch.from_numpy(env.reset()).to(device)
    # Exercise clipping as well as ordinary deterministic and sampled actions.
    for bias in (0., 4.):
        with torch.no_grad():
            model.policy.action_net.bias.fill_(bias)
        for dtype in (torch.float32, torch.float64):
            observed = batch.to(dtype)
            lower, upper = observed.new_tensor(env.lower), observed.new_tensor(env.upper)
            for deterministic in (True, False):
                torch.manual_seed(81)
                original = model.predict(observed.cpu().numpy(), deterministic=deterministic)[0]
                expected = lower+(torch.as_tensor(original, device=device)+1.)*(upper-lower)/2.
                torch.manual_seed(81)
                actual = sb3_controller(model, deterministic=deterministic)(observed, None, 0, config)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    env.close()
