"""Hybrid discrete/continuous hedging with the HPO mixed-gradient estimator.

Paper: Alvo, Russo and Kanoria, Hybrid Policy Optimization,
https://arxiv.org/abs/2605.14297.
Upstream: https://github.com/MatiasAlvo/hybrid-rl/tree/e48ae86da1e8f14c93cbb56e48d87f8674228659
Source mechanism: src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py.
Implementation notes: docs/baseline-methods.md.

This finance adaptation preserves the live-history score plus pathwise first
update, followed by categorical PPO on detached histories. HOLD/TRADE decisions,
continuous sizing and terminal risk use the common financial ledger; donor
inventory features, GAE defaults and the LQR experiment are not imported.
"""
import time
from dataclasses import asdict

import torch
from torch import nn
from torch.distributions import Categorical

from hedging_gym.baselines._shared.checkpoints import (
    check_resume_options,
    due_checkpoint,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from hedging_gym.baselines._shared.controllers import (
    policy_controller as make_controller,
)
from hedging_gym.baselines._shared.policy import (
    BUY,
    HOLD,
    SELL,
    PolicyAction,
    _bounds,
    _ConfiguredPolicy,
    _network,
)
from hedging_gym.baselines._shared.training import _report, _sync
from hedging_gym.environment.finance import bank_subset, bank_to
from hedging_gym.environment.gym_env import TensorHedgingEnv

__all__ = ["HybridPolicy", "hybrid_rollout", "train_hybrid", "make_controller"]


class HybridPolicy(_ConfiguredPolicy):
    """Joint HOLD/TRADE choices and mode-specific continuous target holdings.

    The categorical action contains one bit per instrument. This explicit
    2**n_assets construction is suited to the benchmark's small hedge set, not
    a claim of scalable discrete enumeration for arbitrary portfolios.
    """
    def __init__(self, config, hidden=(64, 64)):
        super().__init__(config)
        if config.execution.holding_lower is None or config.execution.holding_upper is None:
            raise ValueError("this HPO sizing parameterization requires finite holding bounds")
        self.n_modes = 2 ** self.n_assets
        self.discrete = _network(self.feature_dim, self.n_modes, hidden)
        self.continuous = _network(self.feature_dim, self.n_modes * self.n_assets, hidden)
        self.value = _network(self.feature_dim + 1, 1, hidden)
        self.register_buffer("zeta", torch.zeros(()))
        bits = torch.arange(self.n_modes)[:, None] & (1 << torch.arange(self.n_assets))
        self.register_buffer("trade_mask", bits != 0)

    def candidates(self, features, holdings, lower, upper):
        lo, hi = _bounds(features, holdings, lower, upper, self.feature_dim, self.n_assets)
        raw = self.continuous(features).reshape(-1, self.n_modes, self.n_assets)
        targets = lo[:, None] + (hi - lo)[:, None] * (raw.tanh() + 1) / 2
        return torch.where(self.trade_mask[None], targets, holdings[:, None])

    def forward(self, features, holdings, lower, upper, *, deterministic=True, generator=None):
        distribution = Categorical(logits=self.discrete(features))
        indices = (distribution.probs.argmax(-1) if deterministic else
                   torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1))
        candidates = self.candidates(features, holdings, lower, upper)
        target = candidates[torch.arange(len(features), device=features.device), indices]
        modes = torch.where(target > holdings, BUY, torch.where(target < holdings, SELL, HOLD))
        return PolicyAction(target, modes, distribution.log_prob(indices),
                            distribution.probs, distribution.entropy())

    def value_cost(self, observed, zeta):
        inputs = torch.cat((observed, zeta.detach().expand(*observed.shape[:-1], 1)), -1)
        return self.value(inputs).squeeze(-1)


def hybrid_rollout(policy, bank, *, generator=None, deterministic=False):
    """Live accounting histories preserve the HPO cross term.

    Detaching the terminal loss in the score coefficient is necessary;
    detaching the observations in this first rollout would instead remove a
    real gradient through earlier sizing decisions into later mode choices.
    """
    policy.check_config(bank.config)
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    histories, indices, scores, entropies = [], [], [], []
    for _ in range(bank.config.n_decisions):
        histories.append(observed)
        distribution = Categorical(logits=policy.discrete(observed))
        mode = (distribution.probs.argmax(-1) if deterministic else
                torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1))
        candidates = policy.candidates(observed, env.state.positions,
            bank.config.execution.holding_lower, bank.config.execution.holding_upper)
        target = candidates[torch.arange(len(observed), device=observed.device), mode]
        indices.append(mode)
        scores.append(distribution.log_prob(mode))
        entropies.append(distribution.entropy())
        observed, _, _, _, result = env.step(target)
    return result, torch.stack(histories), torch.stack(indices), torch.stack(scores), torch.stack(entropies)


def train_hybrid(train_bank, *, seed=7, updates=8, batch_size=32, hidden=(32, 32),
                 learning_rate=1e-3, zeta_learning_rate=3e-4, ppo_epochs=4,
                 clip_ratio=.2, entropy_coefficient=.001, device="cpu", progress=True,
                 checkpoint_path=None, checkpoint_every=200, resume_from=None):
    """Train categorical exploration and conditional pathwise sizing jointly.

    Gamma=lambda=1 gives complete Monte Carlo returns and zero value at expiry.
    No straight-through gate or silent lot rounding is used. Additional PPO
    epochs update scores/value only: continuous histories are now fixed data.
    Optimizer settings are recorded, not represented as the author's defaults.
    """
    if min(updates, batch_size, ppo_epochs, checkpoint_every) < 1 or min(learning_rate, zeta_learning_rate) <= 0:
        raise ValueError("training budgets and learning rates must be positive")
    config = train_bank.config
    if any(config.execution.vector("minimum_trade", config.n_assets)
           + config.execution.vector("trade_lot", config.n_assets)):
        raise ValueError("HPO continuous sizing does not implement lots or minimum-order actions")
    started = time.perf_counter()
    device = torch.device(device)
    torch.manual_seed(seed)
    bank = bank_to(train_bank, device)
    policy = HybridPolicy(config, hidden).to(device=device, dtype=bank.spot.dtype)
    zeta = nn.Parameter(bank.spot.new_zeros(()))
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    threshold_optimizer = torch.optim.Adam([zeta], lr=zeta_learning_rate)
    sampler = torch.Generator(device=device).manual_seed(seed + 100003)
    options = dict(updates=updates, batch_size=batch_size, hidden=list(hidden), ppo_epochs=ppo_epochs,
                   learning_rate=learning_rate, zeta_learning_rate=zeta_learning_rate,
                   clip_ratio=clip_ratio, entropy_coefficient=entropy_coefficient,
                   gamma=1., gae_lambda=1., cross_term=True)
    if progress:
        _report("train_start", method="hpo", seed=seed, device=str(device), options=options,
                expected_episode_rollouts=updates * batch_size, n_modes=policy.n_modes)
    first_step, previous_seconds, history = 0, 0., []
    new_ppo_extra_updates = 0
    if resume_from:
        saved = load_checkpoint(resume_from, method="hpo", config=config)
        check_resume_options(saved, options)
        if saved["seed"] != seed or saved["training_paths"] != len(bank.spot):
            raise ValueError("resume requires the original seed and frozen training bank")
        policy.load_state_dict(saved["policy"])
        with torch.no_grad():
            zeta.copy_(saved["zeta"])
        optimizer.load_state_dict(saved["optimizer"])
        threshold_optimizer.load_state_dict(saved["threshold_optimizer"])
        sampler.set_state(saved["sampler_rng"])
        restore_rng(saved["rng"])
        first_step, history = saved["step"], saved["history"]
        previous_seconds = saved["elapsed_seconds"]
    else:
        with torch.no_grad():
            initial = bank_subset(bank, slice(0, min(1024, len(bank.spot))))
            losses = hybrid_rollout(policy, initial, generator=sampler)[0]["terminal_loss"]
            zeta.copy_(torch.quantile(losses, config.risk.alpha))
    for update in range(first_step + 1, updates + 1):
        indices = torch.randint(len(bank.spot), (batch_size,), device=device, generator=sampler)
        result, observed, modes, log_probs, entropies = hybrid_rollout(
            policy, bank_subset(bank, indices), generator=sampler)
        loss = result["terminal_loss"]
        risk_cost = config.risk.loss(loss, zeta.detach())
        fixed_observed, old_log_probs = observed.detach(), log_probs.detach()
        returns = risk_cost.detach().expand(config.n_decisions, -1)
        value = policy.value_cost(fixed_observed, zeta)
        advantage = (returns - value).detach()
        # Sum score contributions over time; pathwise terminal loss occurs once.
        score_loss = (log_probs * advantage).sum(0).mean()
        objective = risk_cost.mean() + score_loss + .5 * (value - returns).square().mean()
        objective = objective - entropy_coefficient * entropies.mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        extra_epochs = 0
        for _ in range(ppo_epochs - 1):
            distribution = Categorical(logits=policy.discrete(fixed_observed))
            new_scores = distribution.log_prob(modes)
            ratio = (new_scores - old_log_probs).exp()
            # Cost minimization uses max, the sign-reversed PPO reward objective.
            surrogate = torch.maximum(ratio * advantage,
                ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage).sum(0).mean()
            value = policy.value_cost(fixed_observed, zeta)
            extra_loss = surrogate + .5 * (value - returns).square().mean()
            extra_loss = extra_loss - entropy_coefficient * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            extra_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            extra_epochs += 1
            with torch.no_grad():
                approximate_kl = ((ratio - 1) - (new_scores - old_log_probs)).mean()
            if approximate_kl > .015:
                break
        if config.risk.objective == "es":
            threshold_loss = config.risk.loss(loss.detach(), zeta).mean()
            threshold_optimizer.zero_grad(set_to_none=True)
            threshold_loss.backward()
            threshold_optimizer.step()
        new_ppo_extra_updates += extra_epochs
        with torch.no_grad():
            policy.zeta.copy_(zeta)
        if update == 1 or update % 20 == 0 or update == updates:
            _sync(device)
            elapsed = time.perf_counter() - started
            record = dict(completed=update, total=updates, elapsed_seconds=elapsed,
                          eta_seconds=elapsed * (updates - update) / (update-first_step),
                          batch_risk_loss=float(risk_cost.detach().mean()),
                          zeta=float(zeta.detach()), extra_ppo_epochs=extra_epochs)
            history.append(record)
            if progress:
                _report("train_progress", method="hpo", **record)
        if checkpoint_path and due_checkpoint(update, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method="hpo", config=asdict(config),
                seed=seed, options=options, step=update, policy=policy.state_dict(),
                optimizer=optimizer.state_dict(), threshold_optimizer=threshold_optimizer.state_dict(),
                zeta=zeta.detach(), rng=rng_state(), sampler_rng=sampler.get_state(),
                history=history, training_paths=len(bank.spot),
                elapsed_seconds=previous_seconds+time.perf_counter()-started))
    _sync(device)
    return policy, dict(method="hpo", seed=seed, device=str(device), options=options,
        zeta=float(zeta.detach()), history=history, total_seconds=previous_seconds+time.perf_counter() - started,
        resumed_from=str(resume_from) if resume_from else None, resumed_step=first_step,
        new_rollout_updates=updates-first_step, new_ppo_extra_updates=new_ppo_extra_updates,
        new_policy_optimizer_updates=updates-first_step+new_ppo_extra_updates,
        expected_episode_rollouts=updates * batch_size, initialization_paths=min(1024, len(bank.spot)),
        source="MatiasAlvo/hybrid-rl@e48ae86da1e8f14c93cbb56e48d87f8674228659",
        scope="HPO estimator/PPO adaptation; complete-return ES, not native LQR reproduction",
        training_action_selection="sampled categorical modes; deterministic mode-specific sizes")
