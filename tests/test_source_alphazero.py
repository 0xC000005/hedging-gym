"""The source loop sees only causal states and the common financial ledger."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hedging_gym import benchmark_config, config_from_dict, finance, TimeGrid
from methods.source_alphazero import SourceHedgingGame


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
    path = Path(__file__).resolve().parents[1]/"experiments"/"configs"/f"szehr-{market}-stock.json"
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
