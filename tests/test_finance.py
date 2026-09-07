"""Independent price, cash-ledger and pathwise derivative checks."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from hedging_gym import finance


def test_heston_prices_and_delta_match_quantlib_in_short_and_rare_states():
    config = finance.common_config()
    states = [(1., .04, 30, 1.), (1., .04, 60, .95), (1., .04, 90, 1.05),
              (1., .04, 1, 1.), (.7, .002, 1, 1.), (1., .0001, 1, 1.),
              (.9, .001, 30, 1.05), (1.3, .2, 1, 1.), (1., .005, 30, 1.),
              (.6, .00001, 1, 1.), (1.5, .001, 1, 1.), (1., 1e-8, 1, 1.)]
    candidate = finance.call_price(
        torch.tensor([s for s, v, n, k in states], dtype=torch.float64),
        [v for s, v, n, k in states], [n / 252 for s, v, n, k in states],
        [k for s, v, n, k in states], config).numpy()
    reference = [finance.quantlib_call_price(s, v, n / 252, k, config)
                 for s, v, n, k in states]
    np.testing.assert_allclose(candidate, reference, atol=1e-8, rtol=1e-7)
    assert finance.call_price(torch.tensor(1.2), .04, 0., 1., config).item() == pytest.approx(.2)
    spot = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    delta, = torch.autograd.grad(finance.call_price(spot, .04, 30 / 252, 1., config), spot)
    bump = 1e-5
    ql_delta = (finance.quantlib_call_price(1 + bump, .04, 30 / 252, 1., config)
                - finance.quantlib_call_price(1 - bump, .04, 30 / 252, 1., config)) / (2 * bump)
    assert delta.item() == pytest.approx(ql_delta, abs=1e-8, rel=1e-7)


def test_ledger_matches_numpy_cash_and_derivative_with_real_tickets():
    config = replace(finance.HestonConfig(), n_steps=3, quadratic=(.002, .003, .003),
                     fixed_ticket=(.0001, .0001, .0001))
    bank = finance.generate_market_bank(config, 3, 19, dtype=torch.float64)
    positions = torch.tensor([[[.4, .1, -.1], [.4, .1, -.1], [.7, 0., .2]],
                              [[0., 0., 0.], [.3, -.2, .2], [.2, -.3, .1]],
                              [[.1, -.1, 0.], [.2, -.2, 0.], [.5, -.1, 0.]]], dtype=torch.float64)
    actual = finance.ledger_from_positions(bank, positions)
    expected = finance.numpy_ledger(bank.marks.numpy(), positions.numpy(),
        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), config)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-15, rtol=2e-14)
    assert actual["tickets"].tolist() == [8, 9, 8]
    assert bool((actual["liquidation_cost"] > 0).all())
    variable = positions.clone().requires_grad_()
    gradient, = torch.autograd.grad(finance.ledger_from_positions(bank, variable)["terminal_loss"].sum(), variable)
    plus, minus = positions.clone(), positions.clone()
    plus[0, 2, 0] += 1e-5
    minus[0, 2, 0] -= 1e-5
    # Finite difference uses the separately implemented NumPy cash loop.
    def reference_loss(holdings):
        return finance.numpy_ledger(bank.marks.numpy(), holdings.numpy(),
            bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), config)["terminal_loss"].sum()
    fd = (reference_loss(plus) - reference_loss(minus)) / 2e-5
    assert float(gradient[0, 2, 0]) == pytest.approx(fd, abs=2e-10, rel=2e-8)
    zero_trade = torch.zeros((1, 3), dtype=torch.float64)
    assert finance.transaction_cost(zero_trade, bank.marks[:1, 0], config).item() == 0
    zero_trade[0, 0] = 1e-12
    assert finance.transaction_cost(zero_trade, bank.marks[:1, 0], config).item() >= .0001


def test_causal_features_legal_maps_and_unhedged_payoff():
    config = finance.common_config(n_steps=2)
    bank = finance.generate_market_bank(config, 4, 7, dtype=torch.float64)
    state = finance.initial_state(bank)
    original = finance.observation(bank, 0, state)
    assert original.shape == (4, len(finance.observation_fields(config)))
    changed = finance.MarketBank(bank.spot.clone(), bank.variance.clone(), bank.marks.clone(),
                                 bank.liability.clone(), config)
    changed.spot[:, 1:] *= 3
    changed.variance[:, 1:] *= 4
    changed.marks[:, 1:] *= 5
    changed.liability[:, 1:] *= 6
    torch.testing.assert_close(finance.observation(changed, 0, state), original)
    actions = finance.map_action(torch.tensor([[-1e4, 1e4]], dtype=torch.float64), config)
    assert bool((actions >= torch.tensor(config.holding_lower)).all())
    assert bool((actions <= torch.tensor(config.holding_upper)).all())
    unhedged = finance.ledger_from_positions(bank, torch.zeros((4, 2, 2), dtype=torch.float64))
    torch.testing.assert_close(unhedged["terminal_loss"], bank.liability[:, -1] - bank.liability[:, 0])
    marks, liability = finance.mark_state(bank.spot[:, -1], bank.variance[:, -1], config.n_steps, config)
    torch.testing.assert_close(liability, (bank.spot[:, -1] - config.liability_strike).clamp_min(0))
    assert bool((marks[:, 1:] > 0).all())
