"""Categorical terminal-risk updates with frozen continuous sizing.

References: Expected Policy Gradients, https://jmlr.org/papers/v21/18-012.html;
Hybrid Policy Optimization, https://arxiv.org/abs/2605.14297.
Policy source: https://github.com/MatiasAlvo/hybrid-rl/tree/e48ae86da1e8f14c93cbb56e48d87f8674228659
Implementation notes: docs/counterfactual-update.md.

This local extension compares an all-mode conditional expectation with a
sampled score-function update. It reuses the HPO policy parametrization, not
the full mixed-gradient/PPO update. Market suffixes are training data; deployed
policies receive current state only.
"""
import math
import time
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path

import torch

from hedging_gym.baselines._shared.checkpoints import save_checkpoint
from hedging_gym.baselines._shared.training import _report, _sync
from hedging_gym.environment.finance import LedgerState, bank_subset, bank_to
from hedging_gym.environment.gym_env import TensorHedgingEnv


def mode_loss(logits, terminal_losses, zeta, risk, *, baseline=None, mode_indices=None):
    """Mean root-mode cost surrogate; the caller multiplies by the horizon.

    All-mode costs have shape [B,M]. Sampled costs/mode indices have shape
    [B,1]. Costs, baseline and global threshold are fixed labels: gradients flow
    only through logits. A baseline must be independent of the chosen mode.
    """
    threshold = torch.as_tensor(zeta, device=logits.device, dtype=logits.dtype).detach()
    costs = risk.loss(terminal_losses.detach(), threshold).detach()
    if mode_indices is None:
        if costs.shape != logits.shape:
            raise ValueError("all-mode costs must match [batch,n_modes] logits")
        if baseline is not None:
            costs = costs - baseline.detach().reshape(-1, 1)
        return (logits.softmax(-1) * costs).sum(-1).mean()
    if costs.shape != (len(logits), 1) or mode_indices.shape != costs.shape:
        raise ValueError("sampled costs and mode indices must be [batch,1]")
    if baseline is not None:
        costs = costs - baseline.detach().reshape(-1, 1)
    scores = logits.log_softmax(-1).gather(1, mode_indices)
    return (scores * costs).mean()


def _modes(probabilities, uniforms):
    # Inverse-CDF coupling: each branch gets the same uniform, but its own mode
    # probabilities. Sharing sampled mode indices would change the policy.
    return (uniforms[:, None] >= probabilities.cumsum(-1)).sum(-1).clamp_max(
        probabilities.shape[-1] - 1)


def _target(policy, observed, state, config, uniform=None, mode=None, return_score=False):
    logits = policy.discrete(observed) if mode is None or return_score else None
    if mode is None:
        mode = _modes(logits.softmax(-1), uniform)
    candidates = policy.candidates(observed, state.positions,
        config.execution.holding_lower, config.execution.holding_upper)
    target = candidates[torch.arange(len(observed), device=observed.device), mode]
    if return_score:
        return target, logits.log_softmax(-1).gather(-1, mode[:, None]).squeeze(-1)
    return target


@torch.no_grad()
def counterfactual_rollout(policy, bank, *, time_index, algorithm="all_mode",
                           generator=None, uniforms=None, retain_tape=False):
    """Visit a current-policy prefix, intervene once, and settle every branch.

    Return observed [B,F], mode_indices/losses [B,W], probabilities [B,M],
    work counts, and optionally positions [B,W,T,A] plus uniforms [B,T]. W is
    M for all_mode and 1 for sampled. Root uniforms are ignored when enumerating.
    No future holdings or observations are copied between branches.
    """
    return _counterfactual_rollout(policy, bank, time_index=time_index, algorithm=algorithm,
        generator=generator, uniforms=uniforms, retain_tape=retain_tape)


def _counterfactual_rollout(policy, bank, *, time_index, algorithm="all_mode",
                            generator=None, uniforms=None, retain_tape=False,
                            live_history=False, retain_scores=False):
    """Shared branching engine; joint updates retain history and sizing graphs."""
    config, batch = bank.config, len(bank.spot)
    if config.time_grid.trade_at_maturity:
        raise ValueError("counterfactual updates do not support trading at maturity")
    if algorithm not in ("all_mode", "sampled") or not 0 <= time_index < config.n_steps:
        raise ValueError("choose all_mode/sampled and a decision date before expiry")
    if any(config.execution.vector("minimum_trade", config.n_assets)
           + config.execution.vector("trade_lot", config.n_assets)):
        raise ValueError("frozen continuous sizing does not support lots or minimum orders")
    policy.check_config(config)
    if uniforms is None:
        uniforms = torch.rand((batch, config.n_steps), device=bank.spot.device,
                              dtype=bank.spot.dtype, generator=generator)
    if uniforms.shape != (batch, config.n_steps) or bool(((uniforms < 0) | (uniforms >= 1)).any()):
        raise ValueError("uniforms must be [batch,n_steps] values in [0,1)")
    env = TensorHedgingEnv(bank)
    observed = env.reset()
    prefix, prefix_observed, prefix_scores = [], [], []
    for date in range(time_index):
        target = _target(policy, observed, env.state, config, uniform=uniforms[:, date],
                         return_score=retain_scores)
        if retain_scores:
            target, score = target
            prefix_observed.append(observed)
            prefix_scores.append(score)
        if retain_tape:
            prefix.append(target.clone())
        observed, _, _, _, _ = env.step(target)
    root_observed = observed.clone() if live_history else observed.detach().clone()
    probabilities = policy.discrete(root_observed).softmax(-1)
    if algorithm == "all_mode":
        modes = torch.arange(policy.n_modes, device=observed.device).expand(batch, -1)
    else:
        modes = _modes(probabilities, uniforms[:, time_index])[:, None]
    width = modes.shape[1]
    branch_indices = torch.arange(batch, device=observed.device).repeat_interleave(width)
    branch = TensorHedgingEnv(bank_subset(bank, branch_indices))
    branch.time_index = time_index
    branch.state = LedgerState(*(getattr(env.state, field.name).repeat_interleave(width, 0)
                                 for field in fields(LedgerState)))
    observed = root_observed.repeat_interleave(width, 0)
    suffix, suffix_observed, suffix_scores = [], [], []
    for date in range(time_index, config.n_steps):
        target = _target(policy, observed, branch.state, config,
            mode=modes.reshape(-1) if date == time_index else None,
            uniform=uniforms[:, date].repeat_interleave(width), return_score=retain_scores)
        if retain_scores:
            target, score = target
            suffix_observed.append(observed)
            suffix_scores.append(score)
        if retain_tape:
            suffix.append(target.clone())
        observed, _, _, _, result = branch.step(target)
    output = dict(observed=root_observed, mode_indices=modes.clone(),
        root_probabilities=probabilities, terminal_losses=result["terminal_loss"].reshape(batch, width),
        time_index=time_index, ledger_steps=batch * (time_index + width * (config.n_steps-time_index)),
        terminal_liquidations=batch * width,
        scope="ledger_steps counts decision-date env.step calls; liquidation is reported separately")
    if retain_tape:
        history = [value.repeat_interleave(width, 0) for value in prefix] + suffix
        output.update(positions=torch.stack(history, 1).reshape(batch, width, config.n_steps, config.n_assets),
                      uniforms=uniforms.clone())
    if retain_scores:
        observed_history = [value.repeat_interleave(width, 0) for value in prefix_observed] + suffix_observed
        scores = [value.repeat_interleave(width, 0) for value in prefix_scores] + suffix_scores
        output.update(histories=torch.stack(observed_history).reshape(config.n_steps, batch, width, -1),
                      scores=torch.stack(scores).reshape(config.n_steps, batch, width))
    return output


def train_counterfactual(source_policy, train_bank, *, algorithm, zeta, seed=7,
                         updates=100, batch_size=64, learning_rate=3e-4,
                         device="cpu", progress=True, checkpoint_path=None):
    """Clone a source policy; update its discrete network only, on-policy.

    Both arms use the same seeded uniform intervention-date schedule. Sampled
    batches round upward to match all-mode decision ledger steps at each date;
    complete liquidation work and elapsed time are also reported separately.
    No PPO reuse, critic fitting, threshold updates or continuous gradients occur.
    """
    if algorithm not in ("all_mode", "sampled") or min(updates, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("choose an algorithm and positive training budgets")
    if train_bank.config.time_grid.trade_at_maturity:
        raise ValueError("counterfactual updates do not support trading at maturity")
    if checkpoint_path and Path(checkpoint_path).exists():
        raise FileExistsError("use a new checkpoint path; this pilot does not resume")
    bank = bank_to(train_bank, device)
    policy = deepcopy(source_policy).to(device=device, dtype=bank.spot.dtype)
    policy.eval().requires_grad_(False)
    policy.discrete.requires_grad_(True)
    threshold = torch.as_tensor(zeta, device=device, dtype=bank.spot.dtype).detach().clone()
    if threshold.numel() != 1 or not bool(torch.isfinite(threshold)):
        raise ValueError("zeta must be one finite global training threshold")
    policy.zeta.copy_(threshold)
    frozen = {name: value.detach().clone() for name, value in policy.named_parameters()
              if not name.startswith("discrete.")}
    optimizer = torch.optim.Adam(policy.discrete.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed + 110003)
    date_generator = torch.Generator().manual_seed(seed + 210003)
    options = dict(algorithm=algorithm, updates=updates, batch_size=batch_size,
        learning_rate=learning_rate, fixed_zeta=float(threshold), frozen="continuous and value",
        sampled_budget="ceil(B*(t+M*(T-t))/T) roots; decision-ledger work matched per update",
        gradient_clip=5., intervention="one uniformly sampled date; horizon multiplier",
        hidden=[layer.out_features for layer in policy.discrete if isinstance(layer, torch.nn.Linear)][:-1])
    started, total_steps, total_liquidations, history = time.perf_counter(), 0, 0, []
    if progress:
        _report("counterfactual_start", seed=seed, device=str(device), config=asdict(bank.config),
            options=options, expected_decision_ledger_steps=updates*batch_size*(
                (bank.config.n_steps-1)/2 + policy.n_modes*(bank.config.n_steps+1)/2))
    for step in range(1, updates + 1):
        date = int(torch.randint(bank.config.n_steps, (), generator=date_generator))
        roots = (batch_size if algorithm == "all_mode" else math.ceil(batch_size * (
            date + policy.n_modes * (bank.config.n_steps-date)) / bank.config.n_steps))
        indices = torch.randint(len(bank.spot), (roots,), device=device, generator=generator)
        labels = counterfactual_rollout(policy, bank_subset(bank, indices), time_index=date,
                                       algorithm=algorithm, generator=generator)
        logits = policy.discrete(labels["observed"])
        baseline = policy.value_cost(labels["observed"], threshold).detach()
        objective = bank.config.n_steps * mode_loss(logits, labels["terminal_losses"], threshold,
            bank.config.risk, baseline=baseline,
            mode_indices=labels["mode_indices"] if algorithm == "sampled" else None)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.discrete.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        total_steps += labels["ledger_steps"]
        total_liquidations += labels["terminal_liquidations"]
        _sync(torch.device(device))
        elapsed = time.perf_counter()-started
        record = dict(completed=step, total=updates, time_index=date, roots=roots,
            decision_ledger_steps=labels["ledger_steps"], cumulative_decision_ledger_steps=total_steps,
            cumulative_terminal_liquidations=total_liquidations, elapsed_seconds=elapsed,
            eta_seconds=elapsed*(updates-step)/step, objective=float(objective.detach()),
            gradient_norm=float(norm), mean_branch_ru=float(bank.config.risk.loss(
                labels["terminal_losses"], threshold).mean()))
        history.append(record)
        if progress and (step == 1 or step % 10 == 0 or step == updates):
            _report("counterfactual_progress", algorithm=algorithm, seed=seed, **record)
        if checkpoint_path and (step == 1 or step % 10 == 0 or step == updates):
            save_checkpoint(checkpoint_path, dict(method=algorithm, config=asdict(bank.config),
                seed=seed, step=step, options=options, policy=policy.state_dict(), zeta=threshold.cpu(),
                optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(),
                date_rng=date_generator.get_state(), history=history, elapsed_seconds=elapsed))
    for name, value in policy.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    return policy, dict(method=algorithm, seed=seed, device=str(device), options=options,
        history=history, total_seconds=time.perf_counter()-started,
        decision_ledger_steps=total_steps, terminal_liquidations=total_liquidations,
        training_paths=len(bank.spot), source="HybridPolicy from hedging_gym.baselines.hpo; categorical-only adaptation",
        scope="On-policy categorical credit assignment with frozen sizes/value and global threshold; not full HPO")
