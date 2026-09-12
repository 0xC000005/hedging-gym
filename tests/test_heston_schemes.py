"""Native QuantLib paths and independent conditional-moment checks for QE/QE-M."""

from dataclasses import asdict, replace
import math

import numpy as np
import pytest
import QuantLib as ql
import torch

from hedging_gym import (
    BatesConfig, HestonConfig, benchmark_config, config_from_dict, finance,
)
from hedging_gym.config import market_from_dict
from hedging_gym.paper_benchmarks import buehler_heston, szehr_market


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def _conditional_variance_moments(variance, market, dt):
    decay = math.exp(-market.kappa * dt)
    mean = market.theta + (variance - market.theta) * decay
    variance_of_variance = market.sigma**2 / market.kappa * (
        variance * decay * (1 - decay) + .5 * market.theta * (1 - decay)**2)
    return mean, variance_of_variance / mean**2


def _quantlib_process(market):
    # Fixed reference dates avoid changes to QuantLib's global evaluation date.
    zero = ql.YieldTermStructureHandle(ql.FlatForward(
        ql.Date(2, ql.January, 2024), 0., ql.Actual365Fixed()))
    scheme = (ql.HestonProcess.QuadraticExponentialMartingale
              if market.scheme == "qe_m" else ql.HestonProcess.QuadraticExponential)
    return ql.HestonProcess(zero, zero, ql.QuoteHandle(ql.SimpleQuote(market.spot0)),
                            market.v0, market.kappa, market.theta, market.sigma,
                            market.rho, scheme)


@pytest.mark.parametrize("device", DEVICES)
def test_qe_and_qe_m_paths_match_native_quantlib_on_paper_presets(device):
    # QuantLib v1.43 ql/processes/hestonprocess.cpp: dw[0] is the residual
    # spot normal; dw[1] is the variance normal and its CDF supplies the uniform.
    # The production sampler may draw its unused variance channel independently.
    cdf = ql.CumulativeNormalDistribution()
    for preset in (buehler_heston(), szehr_market("heston")):
        normals = np.random.default_rng(413).normal(size=(preset.n_steps, 4, 2))
        normals[0, :, 1] = [.2, -1., -1., 2.7]
        shocks = np.stack((normals[..., 1],
                           np.vectorize(cdf)(normals[..., 1]), normals[..., 0]), axis=-1)
        tensor_shocks = torch.as_tensor(shocks, dtype=torch.float64, device=device)
        initial = np.array([[preset.market.spot0, variance]
                            for variance in (.04, .0001, 0., 0.)])
        for scheme in ("qe", "qe_m"):
            market = replace(preset.market, scheme=scheme)
            process = _quantlib_process(market)
            oracle = initial.copy()
            spot, variance = torch.as_tensor(initial, dtype=torch.float64,
                                             device=device).unbind(-1)
            branches, saw_zero = set(), False
            for step in range(preset.n_steps):
                branches.update(bool(_conditional_variance_moments(v, market, preset.dt)[1] < 1.5)
                                for v in oracle[:, 1])
                oracle = np.array([list(process.evolve(
                    step * preset.dt, state.tolist(), preset.dt, dw.tolist()))
                    for state, dw in zip(oracle, normals[step])])
                spot, variance = finance.heston_transition(
                    spot, variance, tensor_shocks[step], market, dt=preset.dt)
                actual = torch.stack((spot, variance), dim=-1).cpu().numpy()
                np.testing.assert_allclose(actual, oracle, rtol=2e-12, atol=2e-12,
                                           err_msg=f"{scheme}, {device}, step {step}")
                saw_zero |= bool((oracle[:, 1] == 0).any())
            assert branches == {False, True} and saw_zero


def _quadrature_stock_mean(market, variance, dt):
    """Integrate the actual transition, with no copy of its spot correction.

    Hermite quadrature integrates the two normals in the quadratic branch.
    In the other branch, integrate its zero atom exactly and use Laguerre
    quadrature for the unit exponential tail before integrating the spot normal.
    """
    nodes, normal_weights = np.polynomial.hermite.hermgauss(24)
    normal_nodes, normal_weights = math.sqrt(2) * nodes, normal_weights / math.sqrt(math.pi)
    mean, psi = _conditional_variance_moments(variance, market, dt)
    if psi < 1.5:
        variance_normal = normal_nodes
        variance_uniform = np.full_like(nodes, .5)
        variance_weights = normal_weights
    else:
        probability_zero = (psi - 1) / (psi + 1)
        tail_nodes, tail_weights = np.polynomial.laguerre.laggauss(32)
        # Beyond 30, inverse-CDF uniforms lose precision; omitted probability
        # is below 1e-13, further reduced by the exponential mixture weight.
        keep = tail_nodes < 30
        variance_uniform = np.r_[probability_zero / 2,
            1 - (1 - probability_zero) * np.exp(-tail_nodes[keep])]
        variance_normal = np.zeros_like(variance_uniform)
        variance_weights = np.r_[probability_zero, (1 - probability_zero) * tail_weights[keep]]
    shocks = np.stack((np.repeat(variance_normal, len(nodes)),
                       np.repeat(variance_uniform, len(nodes)),
                       np.tile(normal_nodes, len(variance_uniform))), axis=-1)
    weights = np.outer(variance_weights, normal_weights).ravel()
    spot, next_variance = finance.heston_transition(
        torch.ones(len(weights), dtype=torch.float64),
        torch.full((len(weights),), variance, dtype=torch.float64),
        shocks, market, dt=dt)
    assert float(weights @ next_variance.numpy()) == pytest.approx(mean, abs=2e-12)
    return float(weights @ spot.numpy())


def test_qe_m_conditional_stock_martingale_in_both_variance_branches():
    market, dt = HestonConfig(scheme="qe_m"), 1 / 12
    states = (0., .001, .4)
    assert {_conditional_variance_moments(v, market, dt)[1] < 1.5
            for v in states} == {False, True}
    plain_qe_errors = []
    for variance in states:
        assert _quadrature_stock_mean(market, variance, dt) == pytest.approx(1., abs=2e-12)
        plain_qe_errors.append(abs(_quadrature_stock_mean(
            replace(market, scheme="qe"), variance, dt) - 1))
    # The same integration must detect that removing the correction changes
    # the stock conditional mean; a numerically insensitive fixture is no test.
    assert max(plain_qe_errors) > 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_qe_m_spot_and_variance_gradients_survive_inactive_branches(device):
    # Low positive variance covers both the zero atom and positive tail; zero
    # current variance with a positive tail covers the boundary without asking
    # for the undefined derivative of sqrt(v + v_next) at (0, 0).
    spot = torch.tensor([.9, 1., 1.1, 1.2], dtype=torch.float64,
                        device=device, requires_grad=True)
    variance = torch.tensor([.04, .0001, .0001, 0.], dtype=torch.float64,
                            device=device, requires_grad=True)
    shocks = torch.tensor([[.2, .5, -.3], [-.4, .2, .7], [.4, .999, -.2], [.2, .999, .1]],
                          dtype=torch.float64, device=device)
    market, dt = HestonConfig(scheme="qe_m"), 1 / 365
    next_spot, next_variance = finance.heston_transition(spot, variance, shocks, market, dt=dt)
    assert next_variance[1] == 0 and bool((next_variance[[0, 2, 3]] > 0).all())
    d_spot, d_variance = torch.autograd.grad((next_spot + next_variance).sum(), (spot, variance))
    assert torch.isfinite(d_spot).all() and torch.isfinite(d_variance).all()
    torch.testing.assert_close(d_spot, next_spot / spot, atol=2e-14, rtol=2e-13)
    # Central differences stay within the same branch and away from its atom.
    bump = 1e-7
    with torch.no_grad():
        positive = variance[:3] + bump
        negative = variance[:3] - bump
        plus = finance.heston_transition(spot[:3], positive, shocks[:3], market, dt=dt)
        minus = finance.heston_transition(spot[:3], negative, shocks[:3], market, dt=dt)
        finite_difference = ((plus[0] + plus[1]) - (minus[0] + minus[1])) / (2 * bump)
    torch.testing.assert_close(d_variance[:3], finite_difference, atol=2e-7, rtol=2e-6)


def test_scheme_roundtrip_and_legacy_heston_bates_configs_remain_distinct():
    for market_class in (HestonConfig, BatesConfig):
        assert market_class().scheme == "qe_m"
        for scheme in ("qe", "qe_m"):
            market = market_class(scheme=scheme)
            config = benchmark_config(model=market)
            assert market_from_dict(asdict(market)) == market
            assert config_from_dict(asdict(config)) == config
        explicit = benchmark_config(model=market_class(scheme="qe_m"))
        legacy = asdict(explicit)
        legacy["market"].pop("scheme")
        restored = config_from_dict(legacy)
        assert restored.market == replace(explicit.market, scheme="qe")
        assert market_from_dict(legacy["market"]) == restored.market
        assert restored != explicit
        # Parsing must not relabel or mutate a caller's historical payload.
        assert "scheme" not in legacy["market"]
        with pytest.raises(ValueError, match="scheme"):
            market_class(scheme="unknown")


@pytest.mark.parametrize("device", DEVICES)
def test_bates_without_jumps_preserves_selected_heston_scheme_and_rng(device):
    spot = torch.ones(4, dtype=torch.float64, device=device)
    variance = torch.tensor([.04, .0001, 0., .2], dtype=torch.float64, device=device)
    for scheme in ("qe", "qe_m"):
        bates = BatesConfig(kappa=1., sigma=2., rho=-.7, jump_intensity=0., scheme=scheme)
        heston = HestonConfig(scheme=scheme)
        left = torch.Generator(device=device).manual_seed(79)
        right = torch.Generator(device=device).manual_seed(79)
        expected = finance.transition(spot, variance, config=heston, dt=1 / 365, generator=left)
        actual = finance.transition(spot, variance, config=bates, dt=1 / 365, generator=right)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        assert torch.equal(left.get_state(), right.get_state())
