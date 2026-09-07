"""Focused adapter checks: gradients, updates and authoritative episode tapes."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym.evaluation import evaluate_controller
from hedging_gym.benchmark import benchmark_config
from hedging_gym.finance import (
    generate_market_bank, numpy_ledger, observation_fields,
)
from methods.controllers import classical_controller, policy_controller
from methods.policies import DirectDHPolicy, NoTransactionBandPolicy
from methods.training import rollout, train_policy


@pytest.fixture(scope="module")
def bank():
    config = benchmark_config(model="gbm", n_steps=3)
    return generate_market_bank(config, 24, 1101, dtype=torch.float64)


@pytest.mark.parametrize("policy_class", [DirectDHPolicy, NoTransactionBandPolicy])
def test_policy_terminal_gradient_and_evaluator_tape(bank, policy_class):
    torch.manual_seed(19)
    policy = policy_class(len(observation_fields(bank.config)), bank.config.n_assets, hidden=(8,)).double()
    output = rollout(policy, bank, record_positions=True)
    objective = output["terminal_loss"].mean()
    parameter = policy.continuous[-1].bias
    gradient, = torch.autograd.grad(objective, parameter)
    assert torch.isfinite(gradient).all()
    index = int(gradient.abs().argmax())
    assert gradient[index].abs() > 1e-7
    epsilon = 1e-5
    with torch.no_grad():
        original = parameter[index].clone()
        parameter[index] = original+epsilon
        plus = rollout(policy, bank)["terminal_loss"].mean()
        parameter[index] = original-epsilon
        minus = rollout(policy, bank)["terminal_loss"].mean()
        parameter[index] = original
    torch.testing.assert_close(gradient[index], (plus-minus)/(2*epsilon), rtol=2e-4, atol=1e-8)
    # Shared training/evaluation episodes must execute the same actual trades.
    metrics, tape = evaluate_controller(policy_controller(policy), bank, batch_size=7)
    torch.testing.assert_close(tape["positions"], output["positions"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(tape["terminal_loss"], output["terminal_loss"], rtol=1e-10, atol=1e-12)
    assert metrics["constraint_violations"] == 0
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
                             bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
    for key in ("terminal_loss", "transaction_cost", "turnover", "tickets"):
        np.testing.assert_allclose(tape[key].numpy(), reference[key], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method,policy_class", [("dh", DirectDHPolicy), ("ntb", NoTransactionBandPolicy)])
def test_training_changes_policy_and_frozen_evaluation_is_finite(bank, method, policy_class):
    torch.manual_seed(7)
    initial = policy_class(len(observation_fields(bank.config)), bank.config.n_assets, hidden=(8,)).double()
    policy, metadata = train_policy(method, bank, seed=7, updates=3, batch_size=16,
                                    hidden=(8,), progress=False)
    assert any(not torch.equal(before, after) for before, after in zip(initial.parameters(), policy.parameters()))
    assert metadata["history"][-1]["completed"] == 3
    heldout = generate_market_bank(bank.config, 12, 2201, dtype=torch.float64)
    metrics, _ = evaluate_controller(policy_controller(policy), heldout, zeta=metadata["zeta"])
    assert np.isfinite(metrics["es95"])
    assert np.isfinite(metrics["ru_es95_at_training_zeta"])


@pytest.mark.parametrize("method", ["delta", "delta_gamma", "delta_variance"])
def test_classical_controller_uses_common_evaluation_accounting(bank, method):
    metrics, tape = evaluate_controller(classical_controller(method), bank, batch_size=7)
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
                             bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
    np.testing.assert_allclose(tape["terminal_loss"].numpy(), reference["terminal_loss"], atol=1e-12)
    assert metrics["constraint_violations"] == 0


def test_continuous_training_does_not_silently_project_lot_constraints(bank):
    constrained = replace(bank, config=replace(bank.config, trade_lot=(.01, .1)))
    with pytest.raises(ValueError, match="minimum-trade or lot"):
        train_policy("dh", constrained, updates=1, progress=False)
