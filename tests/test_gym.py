"""Causal scalar/vector Gymnasium behavior and tensor gradients."""

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from hedging_gym import finance, gym_env
from hedging_gym.benchmark import benchmark_config
from hedging_gym.config import ExecutionConfig, RiskConfig, TimeGrid
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.gym_env import HedgingEnv, HedgingVectorEnv, TensorHedgingEnv


@pytest.mark.parametrize("interface", ["tensor", "single", "vector"])
def test_mse_configuration_controls_terminal_reward(interface):
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2),
                              risk=RiskConfig(objective="mse"))
    if interface == "tensor":
        bank = finance.generate_market_bank(config, 2, 809)
        env = TensorHedgingEnv(bank)
        env.reset()
        action = torch.zeros((2, config.n_assets))
    else:
        env = HedgingEnv(config) if interface == "single" else HedgingVectorEnv(2, config)
        env.reset(seed=809)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
    for date in range(config.n_steps):
        _, reward, _, _, info = env.step(action)
        if date < config.n_steps-1:
            assert np.all(np.asarray(reward) == 0)
    np.testing.assert_allclose(reward, -np.asarray(info["terminal_loss"])**2, atol=1e-10)
    if interface != "tensor":
        env.close()


def test_tensor_terminal_accounting_and_gradients_match_independent_cash():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3),
                              execution=ExecutionConfig(fixed_ticket=.0001))
    bank = finance.generate_market_bank(config, 4, 709, dtype=torch.float64)
    positions = torch.tensor([[[.2, .1], [.2, .1], [.4, .2]]], dtype=torch.float64).repeat(4, 1, 1)
    positions.requires_grad_()
    env = TensorHedgingEnv(bank)
    assert env.reset().shape == (4, len(finance.observation_fields(config)))
    for t in range(config.n_steps):
        obs, reward, done, truncated, info = env.step(positions[:, t])
        assert done == (t == config.n_steps - 1) and not truncated
    reference = finance.numpy_ledger(bank.marks.numpy(), positions.detach().numpy(),
        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), config)
    for key in reference:
        np.testing.assert_allclose(info[key].detach().numpy(), reference[key], atol=2e-15)
    tensor_gradient, = torch.autograd.grad(-reward.sum(), positions)
    other = positions.detach().clone().requires_grad_()
    direct_gradient, = torch.autograd.grad(finance.ledger_from_positions(bank, other)["terminal_loss"].sum(), other)
    torch.testing.assert_close(tensor_gradient, direct_gradient)
    assert obs.shape[-1] == len(finance.observation_fields(config))


def test_scalar_gym_api_and_exact_terminal_risk_objective():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3), risk=RiskConfig(alpha=.99))
    env = HedgingEnv(config)
    check_env(env, skip_render_check=True)
    env.close()
    env = HedgingEnv(config, risk_threshold=-.02)
    env.reset(seed=741)
    for _ in range(config.n_steps):
        _, reward, done, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert done
    assert np.isclose(reward, .02 - max(float(info["terminal_loss"]) + .02, 0.) / .01)
    env.close()


def test_refinement_preserves_configuration_and_trading_dates(monkeypatch):
    calls = []
    def generate(config, n_paths, seed, **kwargs):
        calls.append(kwargs["substeps"])
        return finance.generate_market_bank(config, n_paths, seed, **kwargs)
    monkeypatch.setattr(gym_env, "generate_market_bank", generate)
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    for env in (HedgingEnv(config, simulation_substeps=4), HedgingVectorEnv(2, config, simulation_substeps=16)):
        env.reset(seed=421)
        bank = env._tensor_env._bank
        assert bank.config == config and bank.spot.shape[-1] == config.n_steps + 1
        for _ in range(config.n_steps):
            _, _, done, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert np.all(done)
        env.close()
    assert calls == [4, 16]


def test_vector_reset_terminal_info_and_standard_wrapper():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3), risk=RiskConfig(alpha=.90),
                              execution=ExecutionConfig(fixed_ticket=.0001))
    env = HedgingVectorEnv(4, config, risk_threshold=-.02)
    assert isinstance(env, VectorEnv)
    assert env.metadata["autoreset_mode"] == AutoresetMode.DISABLED
    assert env.single_action_space.shape == (2,) and env.action_space.shape == (4, 2)
    obs, _ = env.reset(seed=735)
    assert env.observation_space.contains(obs)
    actions = np.full((4, 2), .1, dtype=np.float32)
    trace = [env.step(actions) for _ in range(config.n_steps)]
    obs, reward, done, truncated, info = trace[-1]
    assert done.shape == (4,) and done.all() and not truncated.any()
    assert np.all(trace[0][1] == 0) and not trace[0][2].any()
    np.testing.assert_allclose(reward, .02 - np.maximum(info["terminal_loss"] + .02, 0.) / .10)
    assert info["_terminal_loss"].all() and info["turnover"].shape == (4, 2)
    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(actions)
    env.reset(seed=735)
    replay = [env.step(actions) for _ in range(config.n_steps)]
    for original, repeated in zip(trace, replay):
        for a, b in zip(original[:4], repeated[:4]):
            np.testing.assert_array_equal(a, b)
    env.reset()
    fresh = [env.step(actions) for _ in range(config.n_steps)][-1]
    assert not np.array_equal(fresh[4]["terminal_loss"], info["terminal_loss"])
    with pytest.raises(ValueError, match="full reset"):
        env.reset(options={"reset_mask": np.array([True, False, False, False])})
    wrapped = RecordEpisodeStatistics(env)
    wrapped.reset(seed=735)
    for _ in range(config.n_steps):
        *_, info = wrapped.step(actions)
    np.testing.assert_array_equal(info["episode"]["l"], config.n_steps)
    np.testing.assert_allclose(info["episode"]["r"], reward)
    wrapped.close()


def test_action_buffer_reuse_cannot_rewrite_previous_holdings():
    config = benchmark_config(time_grid=TimeGrid(n_steps=2),
                              execution=ExecutionConfig(fixed_ticket=.001))
    for env in (HedgingEnv(config), HedgingVectorEnv(1, config)):
        env.reset(seed=23)
        actions = np.full(env.action_space.shape, .1, dtype=np.float32)
        env.step(actions)
        actions[:] = .2
        reused = env.step(actions)[4]
        env.reset(seed=23)
        env.step(np.full(env.action_space.shape, .1, dtype=np.float32))
        independent = env.step(np.full(env.action_space.shape, .2, dtype=np.float32))[4]
        for key in ("terminal_loss", "transaction_cost", "turnover", "tickets"):
            np.testing.assert_array_equal(reused[key], independent[key])
        env.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_vector_tensor_path_preserves_accounting_and_gradients(device):
    config = benchmark_config(time_grid=TimeGrid(n_steps=3),
                              execution=ExecutionConfig(fixed_ticket=.0001))
    env = HedgingVectorEnv(4, config, device=device)
    obs, _ = env.reset_tensor(seed=751)
    assert obs.device.type == device
    positions = torch.full((4, config.n_steps, config.n_assets), .1, device=device, requires_grad=True)
    for t in range(config.n_steps):
        _, reward, done, truncated, info = env.step_tensor(positions[:, t])
    assert done.all() and not truncated.any() and reward.device.type == device
    reference = finance.ledger_from_positions(env._tensor_env._bank, positions)
    torch.testing.assert_close(info["terminal_loss"], reference["terminal_loss"], rtol=0, atol=0)
    got, = torch.autograd.grad(-reward.sum(), positions, retain_graph=True)
    expected, = torch.autograd.grad(reference["terminal_loss"].sum(), positions)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    env.close()


def test_default_wrappers_use_the_benchmark_configuration():
    config = benchmark_config()
    for env in (HedgingEnv(), HedgingVectorEnv(2)):
        assert env.config == config
        observed, _ = env.reset(seed=419)
        assert env.observation_space.contains(observed)
        assert observed.shape[-1] == len(finance.observation_fields(config))
        env.close()


def test_evaluator_uses_configured_risk_and_keeps_fixed_tail_diagnostics():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2),
                              risk=RiskConfig(alpha=.6))
    bank = finance.generate_market_bank(config, 5, 1901, dtype=torch.float64)
    def controller(observed, ledger, time_index, config):
        return torch.zeros_like(ledger.positions)
    metrics, tape = evaluate_controller(controller, bank, batch_size=2, zeta=-.02)
    losses = tape["terminal_loss"].numpy()
    assert metrics["mse"] == pytest.approx(np.mean(losses**2))
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(losses**2)))
    assert metrics["risk_alpha"] == .6
    # Five paths leave two complete observations in the upper 40% tail.
    assert metrics["expected_shortfall"] == pytest.approx(np.sort(losses)[-2:].mean())
    assert metrics["expected_shortfall"] < metrics["es95"]
    assert metrics["ru_at_training_zeta"] == pytest.approx(
        (-.02 + np.maximum(losses + .02, 0) / .4).mean())
    assert metrics["es95"] == pytest.approx(losses.max())
    assert metrics["es99"] == pytest.approx(losses.max())
