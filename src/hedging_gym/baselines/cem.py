"""Cross-entropy rollout planning with fresh conditional market scenarios.

Paper: Rubinstein, The Cross-Entropy Method for Combinatorial and Continuous
Optimization (1999), https://doi.org/10.1023/A:1010091220143.
Implementation: local root-action planner; no external learner is imported.
Implementation notes: docs/baseline-methods.md.

CEM fits a proposal to elite sampled controls. Here it improves the current
hedge, then rolls a frozen feedback policy to expiry. Learned proposals and
optional pathwise refinement are ablations within this planner; it is not MPPI
or AlphaZero, and it does not change the accounting or market simulator.
"""
import time

import torch

from hedging_gym.environment.planning import rollout_branches, sample_continuation

from ._shared.controllers import policy_controller


class RolloutPlanner:
    """Common-random-number CEM over root hedge targets, with feedback thereafter.

    A global training threshold gives the same static terminal ES objective as
    the policies. It is not re-fitted to each node's conditional tail. Search
    sees only current observations; its own conditional scenarios are separate
    from the realized evaluation paths. All candidates share these scenarios.
    """
    def __init__(self, continuation, *, zeta, candidates=16, scenarios=16, iterations=2,
                 seed=9001, guided=False, gradient_steps=0, progress=False):
        if min(candidates, scenarios, iterations) < 1 or gradient_steps < 0:
            raise ValueError("positive search work and nonnegative refinement steps required")
        self.continuation = continuation
        self.zeta = float(zeta)
        self.candidates, self.scenarios, self.iterations = candidates, scenarios, iterations
        self.seed, self.guided, self.gradient_steps = seed, guided, gradient_steps
        self.progress = progress
        self.generator = None
        self.calls = self.rollout_paths = self.price_states = 0
        self.total_seconds = 0.
        self.action_selection = "CEM root-action search with frozen-policy feedback"

    def _score(self, targets, ledger, time_index, config, paths):
        batch, candidates = targets.shape[:2]
        scenarios = self.scenarios
        loss = rollout_branches(targets, ledger, time_index, config, paths,
                                policy_controller(self.continuation, evaluation=False))
        self.rollout_paths += batch * candidates * scenarios
        return config.risk.loss(loss, self.zeta).reshape(batch, candidates, scenarios).mean(-1)

    @torch.no_grad()
    def __call__(self, observed, ledger, time_index, config):
        if observed.device.type == "cuda":
            torch.cuda.synchronize(observed.device)
        started = time.perf_counter()
        self.continuation.check_config(config)
        if any(config.execution.vector("minimum_trade", config.n_assets)
               + config.execution.vector("trade_lot", config.n_assets)):
            raise ValueError("continuous CEM does not implement minimum-order or lot actions")
        if self.generator is None:
            self.generator = torch.Generator(device=observed.device).manual_seed(self.seed)
        if self.progress and time_index == 0:
            print(f"CEM: {len(observed)} roots, {config.n_steps} dates, "
                  f"{self.candidates} candidates × {self.scenarios} scenarios × "
                  f"{self.iterations} iterations; seed {self.seed}", flush=True)
        lower = observed.new_tensor(config.execution.holding_lower)
        upper = observed.new_tensor(config.execution.holding_upper)
        proposal = self.continuation(observed, ledger.positions, lower, upper,
                                     deterministic=True).target_holdings
        anchors = torch.stack((ledger.positions, proposal), 1)
        if self.guided and hasattr(self.continuation, "candidates"):
            anchors = torch.cat((anchors, self.continuation.candidates(
                observed, ledger.positions, lower, upper)), 1)
        if self.candidates <= anchors.shape[1]:
            raise ValueError("candidate budget must exceed the HOLD/policy/mode anchors")
        paths = sample_continuation(observed, time_index, config, self.scenarios, self.generator)
        self.price_states += len(observed) * self.scenarios * (config.n_steps - time_index + 1)
        center = proposal if self.guided else (lower + upper).expand_as(proposal) / 2
        spread = (upper - lower).expand_as(proposal) / 2
        best_targets = proposal
        best_scores = observed.new_full((len(observed),), float("inf"))
        for iteration in range(self.iterations):
            shape = (len(observed), self.candidates - anchors.shape[1], config.n_assets)
            if iteration == 0 and not self.guided:
                samples = lower + (upper - lower) * torch.rand(
                    shape, device=observed.device, dtype=observed.dtype, generator=self.generator)
            else:
                samples = center[:, None] + spread[:, None] * torch.randn(
                    shape, device=observed.device, dtype=observed.dtype, generator=self.generator)
                samples = samples.clamp(lower, upper)
            targets = torch.cat((anchors, samples), 1)
            scores = self._score(targets, ledger, time_index, config, paths)
            elite_count = max(2, self.candidates // 4)
            elite_ids = scores.topk(elite_count, largest=False).indices
            elite = targets.gather(1, elite_ids[..., None].expand(-1, -1, config.n_assets))
            center = elite.mean(1)
            spread = elite.std(1, unbiased=False).clamp_min((upper - lower) * .01)
            win = scores.argmin(1)
            row = torch.arange(len(observed), device=observed.device)
            improved = scores[row, win] < best_scores
            best_targets = torch.where(improved[:, None], targets[row, win], best_targets)
            best_scores = torch.minimum(scores[row, win], best_scores)
            # Retain the incumbent without adding evaluations beyond the budget.
            anchors = anchors.clone()
            anchors[:, 1] = best_targets
        if self.gradient_steps:
            width = upper - lower
            fractions = ((best_targets - lower) / torch.where(width > 0, width, 1.)).clamp(.001, .999)
            latent = torch.logit(fractions).detach().requires_grad_()
            for _ in range(self.gradient_steps):
                with torch.enable_grad():
                    candidate = lower + (upper - lower) * latent.sigmoid()
                    score = self._score(candidate[:, None], ledger, time_index, config, paths)[:, 0]
                    gradient, = torch.autograd.grad(score.sum(), latent)
                better = score.detach() < best_scores
                best_targets = torch.where(better[:, None], candidate.detach(), best_targets)
                best_scores = torch.minimum(best_scores, score.detach())
                latent = (latent - .1 * gradient).detach().requires_grad_()
            # The final updated candidate must be evaluated too.
            candidate = lower + (upper - lower) * latent.detach().sigmoid()
            score = self._score(candidate[:, None], ledger, time_index, config, paths)[:, 0]
            best_targets = torch.where((score < best_scores)[:, None], candidate, best_targets)
        self.calls += 1
        if observed.device.type == "cuda":
            torch.cuda.synchronize(observed.device)
        self.total_seconds += time.perf_counter() - started
        if self.progress and (time_index == config.n_steps - 1 or (time_index + 1) % 10 == 0):
            print(f"CEM date {time_index + 1}/{config.n_steps}; "
                  f"{self.rollout_paths} conditional rollout paths; "
                  f"{self.total_seconds:.1f}s total", flush=True)
        return best_targets.detach()

    def metadata(self):
        return dict(seed=self.seed, candidates=self.candidates, scenarios=self.scenarios,
                    iterations=self.iterations, guided=self.guided, gradient_steps=self.gradient_steps,
                    controller_calls=self.calls, conditional_rollout_paths=self.rollout_paths,
                    conditional_price_states=self.price_states, planning_seconds=self.total_seconds,
                    scope="Root CEM plus frozen feedback to expiry; not full MPPI or AlphaZero")
