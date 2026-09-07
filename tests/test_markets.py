"""Shared market interfaces and bounded numerical checks; no distribution certificate."""

from dataclasses import asdict
import math

import numpy as np
import pytest
import torch

from hedging_gym import finance, benchmark_config, TimeGrid, ExecutionConfig, config_from_dict
from hedging_gym.config import market_from_dict
from hedging_gym.gym_env import HedgingVectorEnv


def quantlib_price(model, spot, variance, days, strike, config, *, tolerance=1e-10, max_evaluations=10000):
    """Independent engines; adaptive Bates integration can be tightened for tiny premiums."""
    import QuantLib as ql
    evaluation = ql.Date(2, ql.January, 2024)
    previous = ql.Settings.instance().evaluationDate
    try:
        ql.Settings.instance().evaluationDate = evaluation
        clock = ql.Business252(ql.NullCalendar())
        zero = ql.YieldTermStructureHandle(ql.FlatForward(evaluation, 0., clock))
        quote = ql.QuoteHandle(ql.SimpleQuote(spot))
        if model == "gbm":
            vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
                evaluation, ql.NullCalendar(), math.sqrt(variance), clock))
            process = ql.BlackScholesMertonProcess(quote, zero, zero, vol)
            engine = ql.AnalyticEuropeanEngine(process)
        else:
            process = ql.BatesProcess(zero, zero, quote, variance, config.kappa,
                config.theta, config.sigma, config.rho, config.jump_intensity,
                config.jump_mean, config.jump_std)
            engine = ql.BatesEngine(ql.BatesModel(process), tolerance, max_evaluations)
        option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike),
                                  ql.EuropeanExercise(evaluation + days))
        option.setPricingEngine(engine)
        return option.NPV()
    finally:
        ql.Settings.instance().evaluationDate = previous


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("model", ["gbm", "bates"])
def test_prices_and_spot_delta_gamma_match_quantlib(model, device):
    config = market_from_dict({"model": model})
    states = [(1., .04, 1, 1.), (1., .04, 30, 1.), (.8, .02, 60, 1.),
              (1.2, .09, 60, 1.), (1., .002, 30, 1.05)]
    candidate = finance.call_price(torch.tensor([x[0] for x in states], dtype=torch.float64, device=device),
        [x[1] for x in states], [x[2] / 252 for x in states], [x[3] for x in states], config)
    oracle = [quantlib_price(model, s, v, days, k, config) for s, v, days, k in states]
    np.testing.assert_allclose(candidate.detach().cpu().numpy(), oracle, atol=1e-8, rtol=1e-7)
    spot = torch.tensor(1., dtype=torch.float64, device=device, requires_grad=True)
    value = finance.call_price(spot, .04, 30 / 252, 1., config)
    delta, = torch.autograd.grad(value, spot, create_graph=True)
    gamma, = torch.autograd.grad(delta, spot)
    bump = 1e-4
    minus, center, plus = [quantlib_price(model, s, .04, 30, 1., config)
                           for s in (1 - bump, 1., 1 + bump)]
    assert float(delta.detach()) == pytest.approx((plus - minus) / (2 * bump), abs=3e-7)
    assert float(gamma.detach()) == pytest.approx((plus - 2 * center + minus) / bump**2, rel=3e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_bates_original_jump_tail_greek_failures_match_tight_quantlib(monkeypatch, device):
    # Original 65,536-path refinement failures: day 17 and day 22, both contracts.
    config = market_from_dict({"model": "bates"})
    states = [(.36072667085931887, .044018864068017205, days) for days in (13, 43)]
    states += [(.30895303443070593, .02160032323403851, days) for days in (8, 38)]
    spot = torch.tensor([x[0] for x in states], dtype=torch.float64, device=device, requires_grad=True)
    variance, maturity = [x[1] for x in states], [x[2] / 252 for x in states]
    def greeks():
        price = finance.call_price(spot, variance, maturity, 1., config)
        delta, = torch.autograd.grad(price.sum(), spot, create_graph=True)
        gamma, = torch.autograd.grad(delta.sum(), spot)
        return price.detach(), delta.detach(), gamma.detach()
    price, delta, gamma = greeks()
    assert (price > 0).all() and (delta > 0).all() and (gamma > 0).all()
    bump = .001  # Tiny premiums require enough bump to avoid cancellation.
    for i, (s, v, days) in enumerate(states):
        q = [quantlib_price("bates", x, v, days, 1., config,
                           tolerance=1e-14, max_evaluations=100000) for x in (s - bump, s, s + bump)]
        assert float(price[i]) == pytest.approx(q[1], abs=5e-13)
        assert float(gamma[i]) == pytest.approx((q[2] - 2 * q[1] + q[0]) / bump**2, rel=.01)
    original = finance._quadrature
    monkeypatch.setattr(finance, "_quadrature", lambda order: original(192))
    _, _, refined_gamma = greeks()
    torch.testing.assert_close(gamma, refined_gamma, rtol=2e-4, atol=1e-11)


@pytest.mark.parametrize("model", ["gbm", "heston", "bates"])
def test_selected_market_prices_and_shared_gym_gradients(model):
    config = benchmark_config(model=model, time_grid=TimeGrid(n_steps=3), execution=ExecutionConfig(fixed_ticket=.0001))
    assert config_from_dict(asdict(config)) == config
    price = finance.call_price(torch.tensor(1., dtype=torch.float64), .04, 30 / 252, 1., config.market)
    assert float(price) == pytest.approx(finance.quantlib_option_price(1., .04, 30 / 252, 1., config.market), abs=1e-8)
    env = HedgingVectorEnv(8, config, simulation_substeps=4)
    observed, _ = env.reset_tensor(seed=19)
    assert observed.shape == (8, len(finance.observation_fields(config)))
    bank = env._tensor_env._bank
    assert bank.spot.shape == (8, 4) and bank.config.dt == config.dt
    actions = torch.full((8, 2), .1, requires_grad=True)
    for _ in range(config.n_steps):
        observed, reward, done, _, info = env.step_tensor(actions)
    assert done.all() and torch.isfinite(torch.autograd.grad(reward.sum(), actions)[0]).all()
    assert torch.isfinite(observed).all()
    positions = np.full((8, config.n_steps, 2), np.float32(.1), dtype=np.float64)
    reference = finance.numpy_ledger(bank.marks.numpy(), positions,
        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), config)
    for key in ("terminal_loss", "transaction_cost", "turnover", "tickets"):
        np.testing.assert_allclose(info[key].detach().numpy(), reference[key], atol=2e-6, rtol=0)
    env.close()


def test_refined_heston_paths_repeat_and_keep_original_trading_grid():
    config = benchmark_config(time_grid=TimeGrid(n_steps=3))
    spot, variance = finance.simulate_market_paths(config.market, config.time_grid, 8, 1967, substeps=4)
    repeated = finance.simulate_market_paths(config.market, config.time_grid, 8, 1967, substeps=4)
    assert spot.shape == variance.shape == (8, 4)
    assert torch.equal(spot, repeated[0]) and torch.equal(variance, repeated[1])
    assert torch.isfinite(spot).all() and torch.isfinite(variance).all()
    assert (spot > 0).all() and (variance >= 0).all()
    torch.testing.assert_close(spot[:, 0], torch.full((8,), config.market.spot0, dtype=torch.float64))
    torch.testing.assert_close(variance[:, 0], torch.full((8,), config.market.v0, dtype=torch.float64))
