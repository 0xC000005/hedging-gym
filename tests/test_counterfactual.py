"""Financial and gradient checks for the categorical-only research prototype."""

import itertools
from copy import deepcopy

import numpy as np
import torch

from hedging_gym.benchmark import benchmark_config, operational_config
from hedging_gym.config import RiskConfig, TimeGrid
from hedging_gym.finance import generate_market_bank, numpy_ledger
from hedging_gym.gym_env import TensorHedgingEnv
from methods.counterfactual import counterfactual_rollout, mode_loss
from methods.hybrid import HybridPolicy


def test_each_counterfactual_has_its_own_tail_cost():
    # The incumbent can be below zeta while another legal action creates a loss.
    # Masking both actions by the incumbent's tail membership loses this signal.
    logits = torch.tensor([[0., 0.]], dtype=torch.float64, requires_grad=True)
    losses = torch.tensor([[-.01, .02]], dtype=torch.float64, requires_grad=True)
    objective = mode_loss(logits, losses, 0., RiskConfig(alpha=.95))
    gradient, target_gradient = torch.autograd.grad(
        objective, (logits, losses), allow_unused=True)
    torch.testing.assert_close(gradient, torch.tensor([[-.1, .1]], dtype=torch.float64))
    assert target_gradient is None


def test_counterfactual_branches_reconcile_with_independent_cash_accounting():
    torch.manual_seed(19)
    config = operational_config(benchmark_config(time_grid=TimeGrid(n_steps=3)),
                                "operational_fixed")
    bank = generate_market_bank(config, 3, 115, dtype=torch.float64)
    policy = HybridPolicy(config, hidden=(8,)).double()
    before = deepcopy(policy.state_dict())
    uniforms = torch.tensor([[.1, .7, .3], [.4, .2, .6], [.8, .3, .9]], dtype=torch.float64)
    branched = counterfactual_rollout(policy, bank, time_index=1,
                                     uniforms=uniforms, retain_tape=True)
    widths = policy.n_modes
    expected = numpy_ledger(
        bank.marks.repeat_interleave(widths, 0).numpy(),
        branched["positions"].reshape(-1, config.n_steps, config.n_assets).numpy(),
        bank.liability[:, 0].repeat_interleave(widths).numpy(),
        bank.liability[:, -1].repeat_interleave(widths).numpy(), config)
    np.testing.assert_allclose(branched["terminal_losses"].reshape(-1).numpy(),
                               expected["terminal_loss"], atol=1e-12)
    sampled = counterfactual_rollout(policy, bank, time_index=1,
                                    algorithm="sampled", uniforms=uniforms)
    chosen = sampled["mode_indices"].squeeze(-1)
    torch.testing.assert_close(sampled["terminal_losses"].squeeze(-1),
                               branched["terminal_losses"][torch.arange(3), chosen])
    for key, value in policy.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        else:
            assert value == before[key]


def test_two_date_counterfactual_gradient_matches_exact_policy_risk():
    """Integrate all common-uniform intervals, not a noisy gradient estimate."""
    torch.manual_seed(23)
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=2))
    bank = generate_market_bank(config, 1, 81, dtype=torch.float64)
    policy = HybridPolicy(config, hidden=(4,)).double()
    policy.continuous.requires_grad_(False)
    parameters = list(policy.discrete.parameters())
    exact_terms, future_boundaries = [], {0., 1.}
    initial = TensorHedgingEnv(bank).reset()
    root_probabilities = policy.discrete(initial).softmax(-1)
    root_boundaries = sorted({0., 1., *root_probabilities.detach().cumsum(-1)[..., :-1].flatten().tolist()})
    for first, second in itertools.product(range(policy.n_modes), repeat=2):
        env = TensorHedgingEnv(bank)
        observed = env.reset()
        probability = bank.spot.new_ones(1)
        for date, mode in enumerate((first, second)):
            probabilities = policy.discrete(observed.detach()).softmax(-1)
            if date == 1:
                future_boundaries.update(probabilities.detach().cumsum(-1)[..., :-1].flatten().tolist())
            probability = probability * probabilities[:, mode]
            target = policy.candidates(observed, env.state.positions,
                config.execution.holding_lower, config.execution.holding_upper)[:, mode]
            observed, _, _, _, result = env.step(target)
        exact_terms.append((probability, result["terminal_loss"]))
    # Put this deterministic test threshold within the enumerated loss range,
    # so the identity is checked with a nonzero tail gradient, not all zeros.
    zeta = torch.stack([loss.detach() for _, loss in exact_terms]).mean()
    exact_objective = sum((probability * config.risk.loss(loss, zeta)).sum()
                          for probability, loss in exact_terms)
    exact = torch.autograd.grad(exact_objective, parameters)

    estimated_terms = []
    # Sampling either date uniformly and multiplying by T is the sum below.
    for date, boundaries in ((0, sorted(future_boundaries)), (1, root_boundaries)):
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            if hi <= lo:
                continue
            uniforms = bank.spot.new_full((1, 2), .5)
            uniforms[0, 1 - date] = (lo + hi) / 2
            batch = counterfactual_rollout(policy, bank, time_index=date, uniforms=uniforms)
            logits = policy.discrete(batch["observed"])
            estimated_terms.append((hi - lo) * mode_loss(
                logits, batch["terminal_losses"], zeta, config.risk))
    estimate = torch.autograd.grad(sum(estimated_terms), parameters)
    for expected, actual in zip(exact, estimate):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    assert any(gradient.abs().max() > 1e-8 for gradient in exact)
