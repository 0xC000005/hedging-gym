"""The source loop sees only causal states and the common financial ledger."""
import json
import os
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from hedging_gym import TimeGrid, benchmark_config, config_from_dict
from hedging_gym.baselines.source_alphazero import SourceHedgingGame, load_source
from hedging_gym.environment import finance


@pytest.mark.parametrize("market", ["gbm", "heston"])
@pytest.mark.parametrize("objective", ["mse", "es"])
def test_source_game_fresh_chance_and_independent_cash(market, objective):
    config = benchmark_config(model=market, time_grid=TimeGrid(n_steps=3), name="operational_fixed")
    game = SourceHedgingGame(config, seed=17, zeta=.01, scale=.1, objective=objective, grid_points=3)
    root = game.getInitState()
    original_cash = root.ledger.cash.clone()
    first, _ = game.getNextState(root, 1, 1)
    second, _ = game.getNextState(root, 1, 1)
    assert not torch.equal(first.spot, second.spot)
    torch.testing.assert_close(root.ledger.cash, original_cash, rtol=0, atol=0)
    state = root
    marks, targets = [root.marks[0]], []
    for action in (1, 1, 2):
        assert game.getGameEnded(state, 1) == 0.
        targets.append(game.targets[action])
        state, _ = game.getNextState(state, 1, action)
        marks.append(state.marks[0])
    _, payoff = finance.mark_state(state.spot, state.variance, state.date, config)
    reference = finance.numpy_ledger(torch.stack(marks)[None].numpy(), torch.stack(targets)[None].numpy(),
                                     original_cash.numpy(), payoff.numpy(), config)
    assert state.loss == pytest.approx(reference["terminal_loss"][0], abs=2e-14)
    if objective == "mse":
        assert game.getGameEnded(state, 1) == pytest.approx(-1-(reference["terminal_loss"][0]/game.scale)**2)
        assert game.getGameEnded(replace(state, loss=0.), 1) == -1.
        assert game.getGameEnded(replace(state, loss=-state.loss), 1) == game.getGameEnded(state, 1)
    else:
        ru = float(config.risk.loss(torch.tensor(state.loss, dtype=torch.float64), game.zeta))
        assert game.getGameEnded(state, 1) == pytest.approx(-1-(ru-game.zeta)/game.scale)
    assert game.getGameEnded(state, 1) <= -1
    assert game.stringRepresentation(state) != game.stringRepresentation(replace(state, loss=state.loss+1e-10))
    assert np.isfinite(game.observe(first)).all()


def test_source_game_does_not_silently_project_state_dependent_constraints():
    with pytest.raises(ValueError, match="action mask"):
        SourceHedgingGame(benchmark_config(name="operational_minimum_trade"), seed=1, zeta=0., scale=1.)


def test_scalar_quantlib_marks_match_common_tensor_pricer():
    config = benchmark_config()
    game = SourceHedgingGame(config, seed=1, zeta=0., scale=.1)
    for s, v, date in ((1., .04, 0), (.7, 0., 29), (1., 1e-8, 29), (1.1, .09, 30)):
        spot, variance = torch.tensor([s], dtype=torch.float64), torch.tensor([v], dtype=torch.float64)
        marks, liability = game.mark(spot, variance, date)
        tensor_marks, tensor_liability = finance.mark_state(spot, variance, date, config)
        torch.testing.assert_close(marks, tensor_marks, atol=1e-8, rtol=1e-7)
        if date == config.n_steps:
            torch.testing.assert_close(liability, tensor_liability, atol=1e-12, rtol=0)
        if v == 0.:
            hedge = config.portfolio.hedges[0]
            near_zero_reference = finance.quantlib_option_price(s, 1e-10,
                (round(hedge.maturity/config.dt)-date)*config.dt, hedge.strike, config.market)
            assert float(marks[0, 1]) == pytest.approx(near_zero_reference, abs=1e-8)


@pytest.mark.parametrize("market", ["gbm", "heston"])
def test_stock_only_source_configuration_runs_complete_shared_ledger(market):
    path = Path(__file__).resolve().parents[1]/"benchmarks"/"configs"/f"szehr-{market}-stock.json"
    config = config_from_dict(json.loads(path.read_text()))
    assert config_from_dict(json.loads(json.dumps(asdict(config)))) == config
    game = SourceHedgingGame(config, seed=41, zeta=0., scale=.1, objective="mse", grid_points=21)
    assert game.targets.shape == (21, config.n_assets) == (21, 1)
    # Source action a corresponds to absolute holdings (a-10)/10, not a trade increment.
    np.testing.assert_allclose(game.targets[:, 0].numpy(), (np.arange(21)-10)/10, atol=1e-15, rtol=0)
    root = state = game.getInitState()
    marks, positions = [root.marks[0]], []
    for date in range(config.n_steps):
        action = 15 if date % 2 == 0 else 10
        positions.append(game.targets[action])
        state, _ = game.getNextState(state, 1, action)
        marks.append(state.marks[0])
        assert (state.loss is not None) == (date == config.n_steps-1)
    payoff = max(float(state.spot[0])-config.portfolio.liability.strike, 0.)
    reference = finance.numpy_ledger(torch.stack(marks)[None].numpy(), torch.stack(positions)[None].numpy(),
                                    root.ledger.cash.numpy(), np.array([payoff]), config)
    assert state.loss == pytest.approx(reference["terminal_loss"][0], abs=2e-14)
    assert state.date*config.dt == config.time_grid.horizon


@pytest.fixture(scope="module")
def native_network_adapter():
    donor_root = os.environ.get("ALPHAZERO_DONOR_ROOT")
    if not donor_root:
        pytest.skip("set ALPHAZERO_DONOR_ROOT to the pinned external donor checkout")
    _, wrapper = load_source(donor_root)
    game = SourceHedgingGame(benchmark_config(), seed=1, zeta=0., scale=.1)
    args = dict(num_channels=8, dropout=.25)
    return game, args, wrapper


@pytest.mark.parametrize("training", [False, True])
def test_source_network_preserves_native_forward_and_raw_value(native_network_adapter, training):
    game, args, wrapper = native_network_adapter
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        adapted = wrapper(game, args).nnet
        with torch.no_grad():
            adapted.fc4.bias.fill_(4.)
        native = type(adapted).__bases__[0](game, args)
        native.lin1 = deepcopy(adapted.lin1)
        native.load_state_dict(adapted.state_dict())
        adapted.train(training)
        native.train(training)
        observed = torch.randn(4, game.feature_dim, requires_grad=True)
        reference_observed = observed.detach().clone().requires_grad_()
        value_inputs = []
        handle = native.fc4.register_forward_pre_hook(
            lambda module, inputs: value_inputs.append(inputs[0]))
        try:
            torch.manual_seed(23)
            reference_policy, bounded_value = native(reference_observed)
        finally:
            handle.remove()
        reference_rng = torch.get_rng_state()
        # Compute only the final linear head independently of the adapter hook.
        reference_value = torch.nn.functional.linear(value_inputs[0], native.fc4.weight, native.fc4.bias)
        torch.manual_seed(23)
        policy, value = adapted(observed)
        assert torch.equal(torch.get_rng_state(), reference_rng)
        assert (value.abs() > 1).any()
        assert not adapted.fc4._forward_hooks
        torch.testing.assert_close(policy, reference_policy, rtol=0, atol=0)
        torch.testing.assert_close(value, reference_value, rtol=0, atol=0)
        torch.testing.assert_close(value.tanh(), bounded_value, rtol=0, atol=0)
        (policy.square().mean() + value.square().mean()).backward()
        (reference_policy.square().mean() + reference_value.square().mean()).backward()
        torch.testing.assert_close(observed.grad, reference_observed.grad, rtol=0, atol=0)
        for name, parameter in adapted.named_parameters():
            torch.testing.assert_close(parameter.grad, native.get_parameter(name).grad, rtol=0, atol=0)
        # Includes batch-normalization running statistics and update counters.
        for name, value in adapted.state_dict().items():
            torch.testing.assert_close(value, native.state_dict()[name], rtol=0, atol=0)


def test_source_network_removes_capture_hook_on_failure(native_network_adapter):
    game, args, wrapper = native_network_adapter
    network = wrapper(game, args).nnet.eval()

    def fail(module, inputs, output):
        raise RuntimeError("injected donor value-head failure")

    failure_handle = network.fc4.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected donor value-head failure"):
            network(torch.zeros(2, game.feature_dim))
        assert list(network.fc4._forward_hooks) == [failure_handle.id]
    finally:
        failure_handle.remove()
    network(torch.zeros(2, game.feature_dim))
    assert not network.fc4._forward_hooks
