"""Chance semantics, source learning loop and common-ledger integration."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym.benchmark import benchmark_config
from hedging_gym.config import EuropeanOption, TimeGrid
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import generate_market_bank, liquidate, mark_state, numpy_ledger
from hedging_gym.gym_env import TensorHedgingEnv
from methods.alphazero import (
    AlphaZeroPolicy, _FinanceSearch, alphazero_controller,
    stochastic_search_batch, train_alphazero,
)


def test_chance_outcomes_are_averaged_and_tree_reaches_terminal_states():
    class CoinModel:
        draws = 0

        def terminal_cost(self, state):
            return state

        def evaluate(self, states):
            return [(np.ones(1), 1.) for _ in states]

        def advance(self, states, actions, rngs):
            output = []
            for _ in states:
                # Exactly balanced sampled outcomes: the search must average
                # rather than treat the favorable zero-cost branch as a move.
                output.append(float(2*(self.draws % 2)))
                self.draws += 1
            return output

    result, = stochastic_search_batch([None], CoinModel(), [np.random.default_rng(9)], simulations=1024)
    assert .7 < result["value"] < 1.3
    assert result["work"]["terminal_evaluations"] > 1

    class ThreeStepModel:
        def terminal_cost(self, state):
            depth, cost = state
            return float(cost) if depth == 3 else None

        def evaluate(self, states):
            return [(np.full(2, .5), float(cost)+(3-depth)*.5) for depth, cost in states]

        def advance(self, states, actions, rngs):
            return [(depth+1, cost+action) for (depth, cost), action in zip(states, actions)]

    result, = stochastic_search_batch([(0, 0)], ThreeStepModel(), [np.random.default_rng(3)],
                                      simulations=256)
    assert result["work"]["maximum_depth"] == 3
    assert result["work"]["terminal_evaluations"] > 0
    assert result["policy"][0] > result["policy"][1]


@pytest.mark.parametrize("extra_option", [False, True])
def test_search_respects_instrument_geometry_constraints_and_ledger(extra_option):
    config = benchmark_config(model="heston", time_grid=TimeGrid(n_steps=3))
    if extra_option:
        config = replace(config, portfolio=replace(config.portfolio,
            hedges=(*config.portfolio.hedges, EuropeanOption(1.1, 8*config.dt, "put"))),
            execution=replace(config.execution, holding_lower=-1., holding_upper=1., proportional=.001))
    config = replace(config, execution=replace(config.execution, holding_lower=-.3, holding_upper=.3,
                                              minimum_trade=.2, trade_lot=.1))
    bank = generate_market_bank(config, 4, 51, dtype=torch.float64)
    policy = AlphaZeroPolicy(config, hidden=(8,), grid_points=3).double()
    controller = alphazero_controller(policy, simulations=8, seed=16)
    env = TensorHedgingEnv(bank)
    observation = env.reset()
    original = env.state.positions.clone()
    _, legal = policy.candidates(env.state.positions, config)
    assert legal[:, :-1].any(-1).all()  # Decimal lots must not collapse the grid to HOLD.
    target = controller(observation, env.state, 0, config)
    torch.testing.assert_close(env.state.positions, original)
    assert target.shape == (4, config.n_assets)
    metrics, tape = evaluate_controller(alphazero_controller(policy, simulations=8, seed=16), bank)
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(), bank.liability[:, 0].numpy(),
                             bank.liability[:, -1].numpy(), config)
    np.testing.assert_allclose(tape["terminal_loss"].numpy(), reference["terminal_loss"], atol=1e-11)
    assert metrics["constraint_violations"] == 0


def test_self_play_fits_policy_and_value_with_shared_terminal_risk():
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 16, 103, dtype=torch.float64)
    torch.manual_seed(7)
    initial = AlphaZeroPolicy(config, hidden=(8,)).double()
    policy, metadata = train_alphazero(bank, updates=2, batch_size=4, hidden=(8,),
                                     simulations=8, gradient_steps=2, progress=False)
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), policy.parameters()))
    assert metadata["optimizer_steps"] == 4
    assert metadata["work"]["terminal_evaluations"] > 0
    assert metadata["history"][-1]["completed"] == 2
    assert all(np.isfinite(record["value_loss"]) for record in metadata["history"])
    heldout = generate_market_bank(config, 8, 203, dtype=torch.float64)
    metrics, _ = evaluate_controller(alphazero_controller(policy, simulations=8), heldout,
                                    zeta=metadata["zeta"])
    assert np.isfinite(metrics["ru_at_training_zeta"])

    # Direct one-step adapter result agrees with the environment at settlement,
    # including a short put liability and transaction/liquidation costs.
    changed = replace(config, portfolio=replace(config.portfolio, liability_quantity=-1.5,
                                              liability=replace(config.portfolio.liability, kind="put")))
    changed_bank = generate_market_bank(changed, 1, 204, dtype=torch.float64)
    env = TensorHedgingEnv(changed_bank)
    obs = env.reset()
    for _ in range(changed.n_steps-1):
        obs, *_ = env.step(torch.zeros_like(env.state.positions))
    model = _FinanceSearch(policy, changed)
    root, = model.roots(obs, env.state, changed.n_steps-1)
    child, = model.advance([root], [len(policy.targets)], [np.random.default_rng(40)])
    _, payoff = mark_state(child.spot, child.variance, changed.n_steps, changed)
    loss = liquidate(child.ledger, child.marks, payoff, changed)["terminal_loss"]
    assert child.terminal_cost == pytest.approx(float(changed.risk.loss(loss, policy.zeta)[0]))
