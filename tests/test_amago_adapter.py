"""Only wrapper accounting/episode semantics; native training qualifies the agent."""

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("amago")

from amago.envs import AMAGOEnv

from hedging_gym.baselines.amago import (
    AmagoController,
    ExplicitEpsilonGreedy,
    MemoryHedgingTask,
)
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import TimeGrid
from hedging_gym.environment.finance import generate_market_bank
from hedging_gym.environment.gym_env import TensorHedgingEnv


def test_memory_sequence_preserves_each_books_cash_and_terminal_ru():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3), name="operational_fixed")
    bank = generate_market_bank(config, 1, 971)
    target = torch.tensor([[.25, .1]])
    reference = TensorHedgingEnv(bank)
    for _ in range(config.n_steps):
        _, _, _, _, raw = reference.step(target)
    expected = -float(config.risk.loss(raw["terminal_loss"], .01)[0])
    wrapped = MemoryHedgingTask((bank,), threshold=.01, episodes=3)
    first, _ = wrapped.reset(seed=17)
    rewards = []
    for step in range(3*config.n_steps):
        observation, reward, done, truncated, info = wrapped.step(target[0].numpy())
        assert not truncated
        assert done == (step == 3*config.n_steps-1)
        if (step+1) % config.n_steps == 0:
            assert reward == pytest.approx(expected)
            assert info["AMAGO_LOG_METRIC terminal_loss"] == pytest.approx(float(raw["terminal_loss"][0]))
            if not done:
                np.testing.assert_array_equal(observation, first)
        else:
            assert reward == 0.
        rewards.append(reward)
    assert sum(rewards) == pytest.approx(3*expected)


def test_native_wrapper_receives_unchanged_observed_market_and_action_bounds():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    changed = replace(config, market=replace(config.market, v0=.09, theta=.09))
    bank = generate_market_bank(changed, 1, 977)
    task = MemoryHedgingTask((bank,), threshold=.01, episodes=2)
    source = AMAGOEnv(task, env_name="qualification")
    wrapper = ExplicitEpsilonGreedy(source)
    timestep, _ = wrapper.reset(seed=23)
    expected = TensorHedgingEnv(bank).reset()[0].numpy()
    np.testing.assert_array_equal(timestep.obs["observation"][0], expected)
    assert wrapper.env_name == "qualification"
    np.testing.assert_array_equal(wrapper.step_count, [0])
    assert source.action_space.shape == (config.n_assets,)
    np.testing.assert_array_equal(source.action_space.low, [-1.] * config.n_assets)
    np.testing.assert_array_equal(source.action_space.high, [1.] * config.n_assets)


def test_vectorized_collector_matches_scalar_books_and_autoresets_only_after_settlement():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3), name="operational_fixed")
    bank = generate_market_bank(config, 1, 983)
    scalar = MemoryHedgingTask((bank,), threshold=.01, episodes=2)
    vector = MemoryHedgingTask((bank,), threshold=.01, episodes=2, num_envs=4)
    initial, _ = scalar.reset(seed=19)
    vector.reset(seed=19)
    for step in range(6):
        obs, reward, done, _, info = scalar.step(np.array([.25, .1], np.float32))
        vobs, vreward, vdone, _, vinfo = vector.step(np.tile([.25, .1], (4, 1)).astype(np.float32))
        np.testing.assert_allclose(vreward, reward, rtol=0., atol=1e-7)
        np.testing.assert_array_equal(vdone, np.full(4, done))
        if step != 5:
            np.testing.assert_allclose(vobs, np.tile(obs, (4, 1)), rtol=0., atol=1e-7)
        else:
            np.testing.assert_allclose(vobs, np.tile(initial, (4, 1)), rtol=0., atol=1e-7)
            np.testing.assert_allclose(vinfo["AMAGO_LOG_METRIC terminal_loss"],
                np.repeat(info["AMAGO_LOG_METRIC terminal_loss"], 4), rtol=0., atol=1e-7)
    assert scalar.completed_books == 2
    assert vector.completed_books == 8


def test_amago_rejects_maturity_decisions_in_training_context_and_query():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=1, trade_at_maturity=True))
    bank = generate_market_bank(config, 1, 991)
    with pytest.raises(ValueError, match="do not support trading at maturity"):
        MemoryHedgingTask((bank,), threshold=.01)
    with pytest.raises(ValueError, match="do not support trading at maturity"):
        AmagoController(None, context_banks=(bank,), risk_threshold=.01)
    controller = AmagoController(None)
    env = TensorHedgingEnv(bank)
    with pytest.raises(ValueError, match="do not support trading at maturity"):
        controller(env.reset(), env.state, 0, config)
