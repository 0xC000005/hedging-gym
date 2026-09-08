"""Task-level learning-potential sampling on an unchanged financial objective.

ACCEL (Parker-Holder et al., ICML 2022) prioritizes high-regret environments and
mutates them. This adapter tests only the first mechanism on a declared finite
task set; it is not an ACCEL reproduction or a task-mutation implementation.
Regret means a gap to a frozen feasible policy, not an oracle. ``regret`` uses
task ES; ``pooled_ru_regret`` aligns the gap to the current shared RU threshold.

The target is pooled terminal ES over a uniform mixture of the declared tasks.
An adaptive task proposal q changes sampling efficiency, not that target: the
entire Rockafellar--Uryasev loss receives p(task)/q(task), including zeta.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import time

import torch

from hedging_gym.evaluation import empirical_es
from hedging_gym.finance import bank_subset, bank_to
from .checkpoints import (check_resume_options, due_checkpoint, load_checkpoint,
                          restore_rng, rng_state, save_checkpoint)
from .training import _report, _sync, rollout


def task_probabilities(scores, *, uniform_mass=.5):
    """Retain support and bound importance weights with a uniform component."""
    scores = torch.as_tensor(scores, dtype=torch.float64, device="cpu").detach().clamp_min(0)
    if scores.ndim != 1 or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("task scores must be a finite nonempty vector")
    if not 0 < uniform_mass <= 1:
        raise ValueError("uniform_mass must lie in (0, 1]")
    uniform = torch.full_like(scores, 1 / len(scores))
    if scores.sum() == 0:
        return uniform
    return uniform_mass * uniform + (1 - uniform_mass) * scores / scores.sum()


def pooled_ru_comparison(student_losses, teacher_losses, zeta, risk):
    """Compare task risks under shared zeta; choose one whole teacher per task."""
    student_ru = risk.loss(student_losses, zeta).mean(-1)
    teacher_ru, teacher_index = risk.loss(teacher_losses, zeta).mean(-1).min(-1)
    return student_ru, teacher_ru, teacher_index


def train_task_curriculum(source_policy, train_banks, *, sampler, teacher_scores,
                          seed=7, updates=300, batch_size=256, learning_rate=3e-4,
                          zeta_learning_rate=3e-4, score_paths=1024, score_every=20,
                          uniform_mass=.5, device="cpu", checkpoint_path=None,
                          checkpoint_every=100, resume_from=None, progress=True,
                          teacher_losses=None):
    """Fine-tune identical source weights with declared task sampling.

    Teacher scores are computed externally on training paths only, by evaluating
    complete causal policies. They never select a different policy per path.
    The fixed score subset is training data, not a validation/final test set.
    Original score selection uses task ES; the pooled-RU correction uses the
    shared threshold. Both optimize the same pooled mixture ES target.
    """
    if sampler not in ("uniform", "stratified", "hard", "regret", "pooled_ru_regret"):
        raise ValueError("unknown task sampler")
    banks = tuple(train_banks)
    if len(banks) < 2 or min(updates, batch_size, score_paths, score_every, checkpoint_every) < 1:
        raise ValueError("at least two tasks and positive work settings are required")
    config = banks[0].config
    for bank in banks:
        if replace(bank.config, market=config.market) != config or len(bank.spot) < score_paths:
            raise ValueError("tasks must share book, clock, execution and risk, with enough score paths")
        source_policy.check_config(bank.config)
    if len(teacher_scores) != len(banks):
        raise ValueError("one feasible reference score is required per task")
    if sampler == "stratified" and batch_size % len(banks):
        raise ValueError("stratified batch size must divide evenly between tasks")
    if sampler == "pooled_ru_regret":
        if (teacher_losses is None or teacher_losses.ndim != 3
                or teacher_losses.shape[0] != len(banks)
                or teacher_losses.shape[2] != score_paths):
            raise ValueError("pooled RU regret requires [tasks, teachers, score_paths] losses")
    started = time.perf_counter()
    device = torch.device(device)
    torch.manual_seed(seed)
    policy = deepcopy(source_policy).to(device=device, dtype=banks[0].spot.dtype)
    banks = tuple(bank_to(bank, device) for bank in banks)
    if teacher_losses is not None:
        teacher_losses = teacher_losses.detach().to(device=device, dtype=banks[0].spot.dtype)
    zeta = torch.nn.Parameter(banks[0].spot.new_zeros(()))
    optimizer = torch.optim.Adam([
        {"params": policy.parameters(), "lr": learning_rate},
        {"params": [zeta], "lr": zeta_learning_rate},
    ])
    index_generator = torch.Generator().manual_seed(seed + 100003)
    options = dict(updates=updates, batch_size=batch_size, learning_rate=learning_rate,
        zeta_learning_rate=zeta_learning_rate, sampler=sampler, score_paths=score_paths,
        score_every=score_every, uniform_mass=uniform_mass, teacher_scores=list(teacher_scores))
    source_configs = [asdict(bank.config) for bank in banks]
    counts = [0] * len(banks)
    proposal = torch.full((len(banks),), 1 / len(banks), dtype=torch.float64)
    history, schedule = [], []
    completed, previous_seconds, score_seconds, score_rollouts = 0, 0., 0., 0
    if resume_from:
        saved = load_checkpoint(resume_from, method="task_curriculum", config=config)
        check_resume_options(saved, options)
        if saved["source_configs"] != source_configs or saved["seed"] != seed:
            raise ValueError("resume requires the saved task configurations and seed")
        if sampler == "pooled_ru_regret" and not torch.equal(
                saved["teacher_losses"], teacher_losses.cpu()):
            raise ValueError("resume requires the original complete-policy teacher loss cache")
        policy.load_state_dict(saved["policy"])
        with torch.no_grad():
            zeta.copy_(saved["zeta"])
        optimizer.load_state_dict(saved["optimizer"])
        index_generator.set_state(saved["index_rng"])
        restore_rng(saved["rng"])
        completed, previous_seconds = saved["step"], saved["elapsed_seconds"]
        proposal, counts = saved["proposal"], saved["task_counts"]
        history, schedule = saved["history"], saved["schedule"]
        score_seconds, score_rollouts = saved["score_seconds"], saved["score_rollouts"]
    else:
        with torch.no_grad():
            initial = [rollout(policy, bank_subset(bank, slice(0, score_paths)))["terminal_loss"]
                       for bank in banks]
            zeta.copy_(torch.quantile(torch.cat(initial), config.risk.alpha))
    _sync(device)
    preparation_seconds = time.perf_counter() - started
    if progress:
        _report("curriculum_start", sampler=sampler, seed=seed, device=str(device),
            task_markets=[asdict(bank.config.market) for bank in banks], options=options,
            target="pooled terminal ES of uniform task mixture", workers=torch.get_num_threads(),
            expected_training_rollouts=updates*batch_size, resumed_step=completed)
    for step in range(completed + 1, updates + 1):
        if sampler in ("hard", "regret", "pooled_ru_regret") and (step - 1) % score_every == 0:
            _sync(device)
            score_start = time.perf_counter()
            policy.eval()
            with torch.no_grad():
                current_losses = torch.stack([rollout(policy, bank_subset(bank, slice(0, score_paths)))[
                    "terminal_loss"] for bank in banks])
                current = [empirical_es(losses, config.risk.alpha) for losses in current_losses]
                reference_details = {}
                if sampler == "pooled_ru_regret":
                    # Minimize expected RU over complete policies, NEVER the
                    # loss per path. Teacher/student use the same global zeta.
                    current_ru, reference_ru, reference_indices = pooled_ru_comparison(
                        current_losses, teacher_losses, zeta, config.risk)
                    scores = (current_ru - reference_ru).cpu().tolist()
                    reference_details = dict(student_ru=current_ru.cpu().tolist(),
                        reference_ru=reference_ru.cpu().tolist(),
                        reference_policy_indices=reference_indices.cpu().tolist(),
                        scoring_zeta=float(zeta.detach()))
                else:
                    scores = (current if sampler == "hard" else
                              [value - reference for value, reference in zip(current, teacher_scores)])
            proposal = task_probabilities(scores, uniform_mass=uniform_mass)
            _sync(device)
            score_seconds += time.perf_counter() - score_start
            score_rollouts += len(banks) * score_paths
            schedule.append(dict(step=step, task_es=current, reference_es=list(teacher_scores),
                scores=scores, probabilities=proposal.tolist(),
                importance_weights=(1 / (len(banks)*proposal)).tolist(), **reference_details))
        policy.train()
        if sampler == "stratified":
            # Keep each bank's market metadata: concatenating banks under one
            # configuration would silently give one task the other's parameters.
            task, weight = -1, 1.
            terms = []
            for index, bank in enumerate(banks):
                indices = torch.randint(len(bank.spot), (batch_size // len(banks),),
                                        generator=index_generator)
                losses = rollout(policy, bank_subset(bank, indices.to(device)))["terminal_loss"]
                terms.append(config.risk.loss(losses, zeta).mean() / len(banks))
                counts[index] += 1
            objective = torch.stack(terms).sum()
        else:
            task = int(torch.multinomial(proposal, 1, generator=index_generator))
            indices = torch.randint(len(banks[task].spot), (batch_size,), generator=index_generator)
            sample = bank_subset(banks[task], indices.to(device))
            losses = rollout(policy, sample)["terminal_loss"]
            weight = 1 / (len(banks) * float(proposal[task]))
            objective = weight * config.risk.loss(losses, zeta).mean()
            counts[task] += 1
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([*policy.parameters(), zeta], 5.,
                                                      error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == updates:
            _sync(device)
            elapsed = previous_seconds + time.perf_counter() - started
            record = dict(step=step, total=updates, task=task, weight=weight,
                probabilities=proposal.tolist(), task_counts=list(counts),
                objective=float(objective.detach()), zeta=float(zeta.detach()),
                gradient_norm_before_clipping=float(gradient_norm),
                elapsed_seconds=elapsed, eta_seconds=(time.perf_counter()-started)
                * (updates-step) / (step-completed))
            history.append(record)
            if progress:
                _report("curriculum_progress", sampler=sampler, seed=seed, **record)
        if checkpoint_path and due_checkpoint(step, updates, checkpoint_every):
            save_checkpoint(checkpoint_path, dict(method="task_curriculum", config=asdict(config),
                source_configs=source_configs, seed=seed, step=step, options=options,
                policy=policy.state_dict(), zeta=zeta.detach(), optimizer=optimizer.state_dict(),
                rng=rng_state(), index_rng=index_generator.get_state(), proposal=proposal,
                task_counts=counts, history=history, schedule=schedule,
                score_seconds=score_seconds, score_rollouts=score_rollouts,
                training_paths=[len(bank.spot) for bank in banks],
                teacher_losses=teacher_losses.cpu() if teacher_losses is not None else None,
                elapsed_seconds=previous_seconds+time.perf_counter()-started))
    _sync(device)
    return policy.eval(), dict(sampler=sampler, seed=seed, options=options,
        classification="Finite-task sampling adaptation, not full ACCEL",
        target="Pooled terminal ES under the uniform task mixture", task_counts=counts,
        zeta=float(zeta.detach()), history=history, schedule=schedule,
        score_seconds=score_seconds, score_rollouts=score_rollouts,
        training_rollouts=updates*batch_size, initialization_rollouts=len(banks)*score_paths,
        preparation_seconds=preparation_seconds,
        total_seconds=previous_seconds+time.perf_counter()-started)
