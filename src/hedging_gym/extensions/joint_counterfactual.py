"""All-mode score gradients combined with HPO's live-history pathwise term.

References: Hybrid Policy Optimization, https://arxiv.org/abs/2605.14297;
Expected Policy Gradients, https://jmlr.org/papers/v21/18-012.html.
Source mechanism: https://github.com/MatiasAlvo/hybrid-rl/blob/e48ae86da1e8f14c93cbb56e48d87f8674228659/src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py
Implementation notes: docs/joint-counterfactual.md.

This local mixed-gradient extension integrates a branching mode and retains
continuous pathwise derivatives. It does not implement an exact conditional-Q
optimizer or claim a new estimator theorem.
"""
import math
import time
from copy import deepcopy
from dataclasses import asdict

import torch

from hedging_gym.baselines._shared.checkpoints import (
    check_resume_options,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from hedging_gym.baselines._shared.training import _report, _sync
from hedging_gym.environment.finance import bank_subset, bank_to

from .counterfactual import _counterfactual_rollout, mode_loss


def joint_objective(policy, bank, *, time_index, zeta, algorithm="all_mode",
                    generator=None, uniforms=None, retain_tape=False, score_scope="trajectory"):
    """Pathwise cost plus live-history scores, integrating one branching mode.

    The categorical coefficients are detached terminal RU costs. The score
    depends on live past continuous actions, preserving the HPO history cross
    term. The all-mode pathwise term averages branches with detached probabilities:
    differentiating those weights here would double count the root score.
    """
    if score_scope not in ("sampled_date", "trajectory"):
        raise ValueError("score_scope must be sampled_date or trajectory")
    branches = _counterfactual_rollout(policy, bank, time_index=time_index,
        algorithm=algorithm, generator=generator, uniforms=uniforms,
        retain_tape=retain_tape, live_history=True, retain_scores=score_scope == "trajectory")
    observed = branches["observed"]
    pathwise_cost = bank.config.risk.loss(branches["terminal_losses"], zeta.detach())
    weights = (branches["root_probabilities"].detach() if algorithm == "all_mode"
               else torch.ones_like(pathwise_cost))
    pathwise = (weights * pathwise_cost).sum(-1).mean()
    if score_scope == "sampled_date":
    # Unbiased single-date estimator; variance can be high.
        baseline = policy.value_cost(observed.detach(), zeta).detach()
        score = bank.config.n_steps * mode_loss(policy.discrete(observed),
            branches["terminal_losses"], zeta, bank.config.risk, baseline=baseline,
            mode_indices=branches["mode_indices"] if algorithm == "sampled" else None)
    else:
        # Full HPO score credit is already available along each branch. Root
        # mode appears once in this sum; no separate root score or T multiplier.
        baseline = policy.value_cost(branches["histories"].detach(), zeta).detach()
        branch_score = (branches["scores"] * (pathwise_cost.detach()[None] - baseline)).sum(0)
        score = (weights * branch_score).sum(-1).mean()
    return pathwise + score, branches, dict(pathwise=pathwise.detach(), score=score.detach())


def train_joint_counterfactual(source_policy, train_bank, *, algorithm, zeta, seed=7,
                              updates=300, batch_size=64, learning_rate=3e-4,
                              device="cpu", progress=True, checkpoint_path=None,
                              resume_from=None, score_scope="trajectory"):
    """Joint sizing/mode learning; fixed threshold and frozen critic in both arms.

    The sampled arm receives more roots to match decision-ledger work. This
    controls estimator differences, not GPU time or backwards FLOPs. Full HPO
    with its critic/PPO/threshold updates is a separate strong comparator.
    """
    if algorithm not in ("all_mode", "sampled") or min(updates, batch_size) < 1:
        raise ValueError("choose all_mode/sampled and positive budgets")
    bank = bank_to(train_bank, device)
    policy = deepcopy(source_policy).to(device=device, dtype=bank.spot.dtype).eval()
    policy.requires_grad_(True)
    policy.value.requires_grad_(False)
    threshold = torch.as_tensor(zeta, device=device, dtype=bank.spot.dtype).detach().clone()
    policy.zeta.copy_(threshold)
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],
                                 lr=learning_rate)
    sampler = torch.Generator(device=device).manual_seed(seed + 110003)
    dates = torch.Generator().manual_seed(seed + 210003)
    options = dict(algorithm=algorithm, updates=updates, batch_size=batch_size,
        learning_rate=learning_rate, fixed_zeta=float(threshold), frozen="value only",
        sampled_budget="ceil(B*(t+M*(T-t))/T) roots; decision-ledger work matched",
        cross_term=True, pathwise=True, gradient_clip=5.,
        score_scope=score_scope,
        intervention="one uniformly sampled branch date; full scores or recorded single-date ablation",
        hidden=[m.out_features for m in policy.discrete if isinstance(m, torch.nn.Linear)][:-1])
    history, first_step, previous_seconds, ledger_steps, liquidations = [], 0, 0., 0, 0
    if resume_from:
        saved = load_checkpoint(resume_from, method="joint_"+algorithm, config=bank.config)
        check_resume_options(saved, options)
        if saved["seed"] != seed or saved["training_paths"] != len(bank.spot):
            raise ValueError("resume requires the original seed and training bank")
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.set_state(saved["sampler_rng"])
        dates.set_state(saved["date_rng"])
        restore_rng(saved["rng"])
        first_step, history = saved["step"], saved["history"]
        previous_seconds = saved["elapsed_seconds"]
        ledger_steps, liquidations = saved["decision_ledger_steps"], saved["terminal_liquidations"]
    started = time.perf_counter()
    if progress:
        _report("joint_counterfactual_start", seed=seed, device=str(device),
            options=options, config=asdict(bank.config), resumed_step=first_step)
    for step in range(first_step + 1, updates + 1):
        date = int(torch.randint(bank.config.n_steps, (), generator=dates))
        roots = batch_size if algorithm == "all_mode" else math.ceil(batch_size * (
            date + policy.n_modes * (bank.config.n_steps-date)) / bank.config.n_steps)
        indices = torch.randint(len(bank.spot), (roots,), device=device, generator=sampler)
        objective, branches, parts = joint_objective(policy, bank_subset(bank, indices),
            time_index=date, zeta=threshold, algorithm=algorithm, generator=sampler,
            score_scope=score_scope)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        ledger_steps += branches["ledger_steps"]
        liquidations += branches["terminal_liquidations"]
        _sync(torch.device(device))
        elapsed = time.perf_counter()-started
        record = dict(completed=step, total=updates, time_index=date, roots=roots,
            elapsed_seconds=previous_seconds+elapsed,
            eta_seconds=elapsed*(updates-step)/(step-first_step),
            decision_ledger_steps=ledger_steps, terminal_liquidations=liquidations,
            objective=float(objective.detach()), gradient_norm=float(norm),
            pathwise=float(parts["pathwise"]), score=float(parts["score"]))
        history.append(record)
        if progress and (step == first_step+1 or step % 25 == 0 or step == updates):
            _report("joint_counterfactual_progress", algorithm=algorithm, seed=seed, **record)
        if checkpoint_path and (step == first_step+1 or step % 25 == 0 or step == updates):
            save_checkpoint(checkpoint_path, dict(method="joint_"+algorithm,
                config=asdict(bank.config), seed=seed, step=step, options=options,
                policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                zeta=threshold.cpu(), sampler_rng=sampler.get_state(), date_rng=dates.get_state(),
                rng=rng_state(), history=history, training_paths=len(bank.spot),
                elapsed_seconds=previous_seconds+elapsed, decision_ledger_steps=ledger_steps,
                terminal_liquidations=liquidations))
    return policy, dict(method="joint_"+algorithm, seed=seed, options=options, history=history,
        total_seconds=previous_seconds+time.perf_counter()-started,
        decision_ledger_steps=ledger_steps, terminal_liquidations=liquidations,
        scope="Joint mixed-gradient development; fixed threshold/critic, no PPO reuse",
        source="HPO live-history gradient + Expected Policy Gradients all-mode integration")
