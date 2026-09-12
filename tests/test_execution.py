"""Execution economics, legal increments and shared evaluation accounting."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym import ExecutionConfig, TimeGrid, benchmark_config
from hedging_gym.environment import finance
from hedging_gym.environment.gym_env import (
    HedgingEnv,
    HedgingVectorEnv,
    TensorHedgingEnv,
)
from hedging_gym.evaluation import empirical_es, evaluate_controller


def execution_config():
    return benchmark_config(time_grid=TimeGrid(n_steps=3), execution=ExecutionConfig(
        minimum_trade=.5, trade_lot=.25, minimum_commission=(.003, .004)))


def test_commission_is_a_floor_and_keeps_smooth_cost_gradients():
    config = benchmark_config(execution=ExecutionConfig(minimum_commission=(.01, .02),
        proportional=.1, quadratic=(.2, .3), fixed_ticket=(.001, .002)))
    trade = torch.tensor([[0., 0.], [.01, -.02], [.5, -.5]], dtype=torch.float64, requires_grad=True)
    costs = finance.transaction_cost(trade, torch.ones_like(trade), config)
    np.testing.assert_allclose(costs.detach(), [0., .03314, .228], atol=1e-15)
    gradient, = torch.autograd.grad(costs.sum(), trade)
    np.testing.assert_allclose(gradient.detach(), [[0., 0.], [.004, -.012], [.3, -.4]], atol=1e-15)


def test_minimum_and_lots_apply_to_increments_with_mandatory_closeout():
    config = benchmark_config(time_grid=TimeGrid(n_steps=2), execution=ExecutionConfig(
        minimum_trade=.3, trade_lot=.1, minimum_commission=.001))
    bank = finance.generate_market_bank(config, 2, 409)
    half = torch.full((2, 2), .5)
    state = finance.trade_step(finance.initial_state(bank), half, bank.marks[:, 0], config)
    assert finance.feasible_targets(half, half, config).all()
    assert not finance.feasible_targets(half, half + .1, config).any()
    assert not finance.feasible_targets(half, half + .31, config).any()
    remaining = torch.full_like(half, .2)
    state = finance.trade_step(state, remaining, bank.marks[:, 1], config)
    with pytest.raises(ValueError, match="minimum-trade"):
        finance.trade_step(state, torch.zeros_like(half), bank.marks[:, -1], config)
    final = finance.liquidate(state, bank.marks[:, -1], bank.liability[:, -1], config)
    torch.testing.assert_close(final["liquidation_cost"],
                               finance.transaction_cost(-remaining, bank.marks[:, -1], config))
    assert (final["tickets"] == 6).all()
    lot_only = replace(config, execution=replace(config.execution, minimum_trade=0.))
    assert not finance.feasible_targets(torch.zeros_like(half), torch.full_like(half, 1e-10), lot_only).any()


def test_tensor_masks_match_execution_and_rejection_preserves_state():
    env = TensorHedgingEnv(finance.generate_market_bank(execution_config(), 2, 193))
    assert env.reset().shape == (2, len(finance.observation_fields(env.config)))
    candidates = torch.tensor([[0., 0.], [.5, .5], [.25, .5], [.6, .5], [3., .5], [float("nan"), 0.]])
    torch.testing.assert_close(env.action_mask(candidates),
        torch.tensor([[True, True, False, False, False, False]] * 2))
    entry = torch.tensor([[.75, .5], [1., .75]])
    env.step(entry)
    torch.testing.assert_close(env.action_mask(torch.stack((entry, entry + .25), dim=1)),
                               torch.tensor([[True, False]] * 2))
    with pytest.raises(ValueError, match="minimum-trade and lot"):
        env.step(entry + .25)
    assert env.time_index == 1
    torch.testing.assert_close(env.state.positions, entry, rtol=0, atol=0)
    lots = TensorHedgingEnv(finance.generate_market_bank(
        benchmark_config(time_grid=TimeGrid(n_steps=3), execution=ExecutionConfig(trade_lot=.1)), 2, 193))
    lots.reset()
    lots.step(torch.full((2, 2), .1))
    mixed = torch.full((2, 2), .2, dtype=torch.float64, requires_grad=True)
    assert bool(lots.action_mask(mixed).all())
    lots.step(mixed)
    assert lots.state.positions.dtype == torch.float32
    gradient, = torch.autograd.grad(lots.state.cash.sum(), mixed)
    assert bool(torch.isfinite(gradient).all())


def test_gym_and_tensor_masks_preserve_terminal_liquidation_exemption():
    scalar, vector = HedgingEnv(execution_config()), HedgingVectorEnv(1, execution_config())
    candidates = np.array([[0., 0.], [.25, .25], [.75, .75]], dtype=np.float32)
    for env in (scalar, vector):
        env.reset(seed=197)
    np.testing.assert_array_equal(scalar.action_mask(candidates), [True, False, True])
    np.testing.assert_array_equal(vector.action_mask(candidates), [[True, False, True]])
    torch.testing.assert_close(vector.action_mask_tensor(torch.from_numpy(candidates)),
                               torch.tensor([[True, False, True]]))
    results = []
    for env in (scalar, vector):
        with pytest.raises(ValueError, match="minimum-trade and lot"):
            env.step(np.full(env.action_space.shape, .25, dtype=np.float32))
        for quantity in (.75, .25, .25):
            _, _, done, truncated, info = env.step(np.full(env.action_space.shape, quantity, dtype=np.float32))
        assert np.all(done) and not np.any(truncated)
        torch.testing.assert_close(env._tensor_env.state.positions, torch.zeros((1, 2)))
        assert np.all(info["liquidation_cost"] >= .0069)
        results.append(info)
        env.close()
    for key in ("terminal_loss", "transaction_cost", "turnover", "tickets"):
        np.testing.assert_array_equal(results[0][key], results[1][key][0])


def test_evaluator_tape_reconciles_and_infeasible_control_is_not_projected():
    config = execution_config()
    bank = finance.generate_market_bank(config, 4, 199, dtype=torch.float64)
    def control(observed, ledger, date, config):
        return torch.full_like(ledger.positions, .75 if date == 0 else .25, dtype=torch.float32)
    metrics, tape = evaluate_controller(control, bank, batch_size=3)
    expected = torch.tensor([.75, .25, .25], dtype=torch.float64)[None, :, None].expand(4, 3, 2)
    torch.testing.assert_close(tape["positions"], expected, rtol=0, atol=0)
    reference = finance.numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), config)
    for key in ("terminal_loss", "transaction_cost", "turnover", "tickets"):
        np.testing.assert_allclose(tape[key], reference[key], atol=2e-14, rtol=0)
    assert metrics["constraint_violations"] == 0
    assert metrics["es95"] == pytest.approx(max(reference["terminal_loss"]))
    with pytest.raises(ValueError, match="minimum-trade and lot"):
        evaluate_controller(lambda observed, ledger, date, config: torch.full_like(ledger.positions, .6), bank)


def test_empirical_es_weights_the_fractional_tail_boundary():
    # Worst 1.5 observations: 10 plus half of 4, divided by 1.5.
    assert empirical_es(torch.tensor([1., 10., -2., 4., 0., 3.]), .75) == pytest.approx(8.)
