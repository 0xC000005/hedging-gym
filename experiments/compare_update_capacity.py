"""Compare context-only and full-network adaptation from identical checkpoints.

The finance benchmark, pretrained weights, selected initial context, ES threshold
and minibatch stream are shared. Only the parameters permitted to adapt differ.
This is a fixed-optimizer diagnostic, not a tuned-method leaderboard.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time

import torch

from experiments.compare_fast_adaptation import (
    TARGET_MARKETS, market_config, rescore_contexts, write_json,
)
from experiments.qualify_adaptation import load_bank
from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import BANK_FIELDS, bank_subset, bank_to, generate_market_bank
from methods.adaptation import AdaptationUpdater, TaskEmbeddedPolicy
from methods.checkpoints import save_checkpoint
from methods.controllers import policy_controller
from methods.training import _report, _sync, rollout


MILESTONES = (0, 10, 50, 200)


def prepare_banks(args):
    directory = args.output / "banks"
    directory.mkdir(parents=True, exist_ok=True)
    requests = [(name, values, role, count, seed + i)
                for i, (name, values) in enumerate(TARGET_MARKETS.items())
                for role, count, seed in (("train", 4096, 970000),
                                          ("cal", 1024, 980000),
                                          ("eval", 8192, 990000))]
    for number, (name, values, role, count, seed) in enumerate(requests, 1):
        config = market_config(values)
        count = 32 if args.smoke else count
        path = directory / f"{name}-{role}.pt"
        if path.exists():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            if (saved["seed"] != seed or len(saved["spot"]) != count
                    or saved["config"] != asdict(config)):
                raise ValueError("existing bank differs; choose a new output directory")
            continue
        _report("bank_start", target=name, role=role, paths=count, seed=seed,
                device=args.device, completed=number-1, total=len(requests))
        started = time.perf_counter()
        bank = generate_market_bank(config, count, seed, device=args.device)
        _sync(torch.device(args.device))
        torch.save(dict(config=asdict(config), seed=seed,
            generation_seconds=time.perf_counter()-started,
            **{key: getattr(bank, key).cpu() for key in BANK_FIELDS}), path)
        _report("bank_complete", completed=number, total=len(requests),
                seconds=time.perf_counter()-started)


def matched_updater(policy, metadata, train, initial, zeta, *, mode, seed,
                    batch_size=256, updates=10):
    """Start both modes at the same policy, threshold and fresh Adam state.

    The general online updater recalibrates the threshold only when starting a
    new embedding task. Here it has already been initialized identically for
    both arms, so mark this market initialized before the first call.
    """
    learner = deepcopy(policy)
    updater = AdaptationUpdater(learner, metadata=metadata, mode=mode, seed=seed,
        updates=updates, batch_size=batch_size, progress=False)
    learner.active_task = None
    learner.source_embeddings.requires_grad_(False)  # Not used by the target policy.
    with torch.no_grad():
        learner.embedding.copy_(initial)
        updater.zeta.copy_(zeta)
    updater.last_market = train.config.market
    return updater


def path_usage(seed, count, batch_size, updates, initialization_paths):
    """Replay indices without consuming the learner's RNG; count data, not draws."""
    generator = torch.Generator().manual_seed(seed+100003)
    seen = torch.zeros(count, dtype=torch.bool)
    seen[:initialization_paths] = True
    for _ in range(updates):
        seen[torch.randint(count, (batch_size,), generator=generator)] = True
    return int(seen.sum())


def compare(args):
    checkpoint = args.checkpoints / "adh" / f"seed-{args.seed}" / "pretrained.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    options = metadata["options"]
    policy = TaskEmbeddedPolicy(config_from_dict(metadata["config"]),
        n_tasks=len(metadata["source_configs"]), embedding_dim=options["embedding_dim"],
        hidden=options["hidden"]).to(args.device)
    policy.load_state_dict(saved["policy"])
    policy.prepare_adaptation()
    budgets = (0, 1, 2) if args.smoke else MILESTONES
    batch_size, chunk = (16, 1) if args.smoke else (256, 10)
    device = torch.device(args.device)
    seed = args.seed+1000
    output = args.output / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "pretraining.json", metadata)
    for target in TARGET_MARKETS:
        banks = {role: load_bank(args.output / "banks" / f"{target}-{role}.pt")
                 for role in ("train", "cal", "eval")}
        train = bank_to(banks["train"], device)
        calibration = bank_to(banks["cal"], device)
        scores, selection_work = rescore_contexts(policy, calibration,
            range(len(policy.source_embeddings)), device)
        index = min(scores, key=lambda row: row[1])[0]
        initial = policy.source_embeddings[index].detach().clone()
        scorer = deepcopy(policy)
        with torch.no_grad():
            scorer.embedding.copy_(initial)
            count = min(1024, len(train.spot))
            started = time.perf_counter()
            losses = rollout(scorer, bank_subset(train, slice(0, count)))["terminal_loss"]
            zeta = torch.quantile(losses, train.config.risk.alpha)
            _sync(device)
            initialization_seconds = time.perf_counter()-started
        selection = dict(selected_context=index, scores=scores,
            calibration_distinct_paths=len(calibration.spot), **selection_work,
            threshold_initialization_paths=count,
            threshold_initialization_seconds=initialization_seconds,
            initial_zeta=float(zeta), initial_embedding=initial.cpu().tolist())
        target_dir = output / target
        target_dir.mkdir(parents=True, exist_ok=True)
        write_json(target_dir / "initialization.json", selection)
        # Alternate execution order across targets/seeds, without changing RNG.
        modes = ("embedding", "finetune")
        if (list(TARGET_MARKETS).index(target)+args.seed) % 2:
            modes = modes[::-1]
        for mode in modes:
            directory = target_dir / mode
            directory.mkdir(parents=True, exist_ok=True)
            updater = matched_updater(policy, metadata, train, initial, zeta,
                mode=mode, seed=seed, batch_size=batch_size, updates=chunk)
            learner = updater.policy
            original_shared = {name: value.detach().clone()
                               for name, value in learner.shared.named_parameters()}
            report = dict(mode=mode, target=target, policy_seed=args.seed,
                config=asdict(train.config), initialization=selection,
                source_pretraining=metadata["options"], minibatch_seed=seed+100003,
                learning_rate=updater.learning_rate, zeta_learning_rate=updater.zeta_learning_rate,
                batch_size=batch_size,
                active_adaptation_parameters=sum(p.numel() for p in learner.parameters()
                                                 if p.requires_grad),
                timing_scope="Adaptation excludes final evaluation and bank generation; parallel jobs may overlap",
                milestones=[])
            snapshot = directory / "latest.pt"
            if snapshot.exists():
                previous = torch.load(snapshot, weights_only=False, map_location="cpu")
                updater.load_state_dict(previous["updater"])
                report = previous["report"]
            complete = {point["updates"] for point in report["milestones"]}
            for budget in budgets:
                if budget in complete:
                    continue
                if updater.completed_steps > budget:
                    raise ValueError("saved checkpoint has passed an unsaved milestone")
                while updater.completed_steps < budget:
                    updater(train)
                metrics, tape = evaluate_controller(policy_controller(learner), banks["eval"],
                    device=device, batch_size=1024, label=f"{mode}/{target}/{budget}")
                changed = [name for name, value in learner.shared.named_parameters()
                           if not torch.equal(value.detach(), original_shared[name])]
                if mode == "embedding" and changed:
                    raise RuntimeError("embedding-only arm changed shared weights")
                record = dict(updates=budget, metrics=metrics,
                    shared_parameters_changed=changed,
                    unique_target_paths=len(calibration.spot)+path_usage(
                        seed, len(train.spot), batch_size, budget, count),
                    gradient_episode_rollouts=budget*batch_size,
                    calibration_episode_rollouts=selection_work["episode_rollouts"],
                    threshold_initialization_paths=count,
                    adaptation_seconds=sum(h["elapsed_seconds"] for h in updater.history),
                    context=learner.embedding.detach().cpu().tolist(), zeta=float(updater.zeta.detach()))
                report["milestones"].append(record)
                torch.save(tape, directory / f"{budget}-tape.pt")
                payload = dict(step=budget, updater=updater.state_dict(), report=report)
                save_checkpoint(directory / f"{budget}-policy.pt", payload)
                save_checkpoint(snapshot, payload)
                write_json(directory / "curve.json", report)
                _report("capacity_point", mode=mode, seed=args.seed, target=target,
                    updates=budget, es95=metrics["es95"],
                    adaptation_seconds=record["adaptation_seconds"],
                    unique_target_paths=record["unique_target_paths"])
    _report("capacity_complete", seed=args.seed, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("banks", "compare"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "compare" and args.checkpoints is None:
        parser.error("compare requires --checkpoints")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    _report("capacity_start", options={k: str(v) if isinstance(v, Path) else v
                                      for k, v in vars(args).items()})
    (prepare_banks if args.stage == "banks" else compare)(args)


if __name__ == "__main__":
    main()
