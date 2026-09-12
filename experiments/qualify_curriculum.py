"""Compare task curricula without changing the benchmark or pooled ES target.

Shared source checkpoints and A-market banks are read only. New B-market banks,
training state, raw evaluation tapes, and compact reports belong outside Git.
This is a finite-task development comparison, not a final ranking or ACCEL run.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from hedging_gym.benchmark import adaptation_configs, operational_config
from hedging_gym.evaluation import empirical_es, evaluate_controller
from hedging_gym.finance import BANK_FIELDS, MarketBank, bank_subset, bank_to, generate_market_bank
from methods.controllers import policy_controller
from methods.curriculum import train_task_curriculum
from methods.hybrid import HybridPolicy
from methods.training import POLICIES, _report, _sync, rollout
from experiments.qualify_policies import load_bank


def load_policy(path, config, device, dtype):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cls = HybridPolicy if saved["method"] == "hpo" else POLICIES[saved["method"]]
    policy = cls(config, hidden=saved["options"]["hidden"]).to(device=device, dtype=dtype)
    policy.load_state_dict(saved["policy"])
    return policy.eval()


def evaluate(policy, banks, output, device, label):
    metrics, tapes = {}, []
    for name, bank in zip(("A", "B"), banks):
        metric, tape = evaluate_controller(policy_controller(policy), bank, device=device,
            batch_size=1024, label=f"{label}/{name}", progress=True)
        torch.save(tape, output / f"{name}-development-tape.pt")
        metrics[name] = metric
        tapes.append(tape["terminal_loss"])
    metrics["pooled_es95"] = empirical_es(torch.cat(tapes), banks[0].config.risk.alpha)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--presets", nargs="+", choices=("basic", "operational_fixed"),
                        default=["basic", "operational_fixed"])
    parser.add_argument("--samplers", nargs="+", choices=("uniform", "stratified", "hard", "regret", "pooled_ru_regret"),
                        default=["uniform", "hard", "regret"])
    parser.add_argument("--student", choices=("dh", "ntb"), default="ntb")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--score-paths", type=int, default=1024)
    parser.add_argument("--score-every", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source, output = args.source_dir.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("write new evidence outside the source artifact directory")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "comparison.json"
    report = (json.loads(report_path.read_text()) if report_path.exists() else dict(
        classification="Finite A/B task curriculum development; not full ACCEL or final evidence",
        arguments={key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        target="Pooled cost-inclusive terminal ES95 under uniform A/B task mixture",
        scientific_limits=["B is available in training for every arm; no unseen-regime claim",
            "Teacher is a finite feasible source-policy ensemble, not optimal regret",
            "Task priorities are training-only; one global zeta fits pooled ES",
            "Matched optimizer updates do not imply matched total compute",
            "Fixed-task selection does not test ACCEL environment mutation"],
        sampler_labels=dict(regret="task_es_regret", pooled_ru_regret="pooled_ru_regret"), presets={}))

    def write_report():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    start = time.perf_counter()
    train_a = load_bank(source / "train_bank.pt", "basic")
    development_a = load_bank(source / "development_bank.pt", "basic")
    config_b = adaptation_configs(train_a.config)[1][1]
    banks_b = []
    for role, count, seed in (("train", len(train_a.spot), 710101),
                               ("development", len(development_a.spot), 710201)):
        path = output / f"B-{role}-bank.pt"
        if path.exists():
            bank = load_bank(path, "basic")
            if bank.config != config_b or len(bank.spot) != count:
                raise ValueError("saved B bank differs from the declared task/count")
        else:
            _report("curriculum_bank_start", role=role, paths=count, seed=seed,
                    device=args.device, config=asdict(config_b))
            bank = bank_to(generate_market_bank(config_b, count, seed, device=args.device), "cpu")
            torch.save(dict(config=asdict(config_b), seed=seed,
                            **{key: getattr(bank, key) for key in BANK_FIELDS}), path)
            _report("curriculum_bank_complete", role=role, paths=count)
        banks_b.append(bank)
    _sync(args.device)
    report.setdefault("shared_bank_preparation_seconds", time.perf_counter()-start)
    report["banks"] = dict(A_train=str(source / "train_bank.pt"),
        A_development=str(source / "development_bank.pt"), B_train_seed=710101,
        B_development_seed=710201, training_paths_per_task=len(train_a.spot),
        development_paths_per_task=len(development_a.spot))
    write_report()
    for preset in args.presets:
        config = operational_config(train_a.config, preset)
        def overlay(bank):
            return MarketBank(*(getattr(bank, key) for key in BANK_FIELDS),
                              operational_config(bank.config, preset))
        train = tuple(overlay(bank) for bank in (train_a, banks_b[0]))
        development = tuple(overlay(bank) for bank in (development_a, banks_b[1]))
        policy_root = source / "policies"
        step = 3000
        if preset != "basic":
            policy_root = policy_root / preset
            step = 1000
        def source_path(method, seed):
            return policy_root / f"{method}-seed{seed}" / f"step-{step}.pt"
        scores, teacher_loss_cache = [], []
        score_banks = [bank_to(bank_subset(bank, slice(0, args.score_paths)), args.device)
                       for bank in train]
        _sync(args.device)
        teacher_started = time.perf_counter()
        for method in ("dh", "ntb", "hpo"):
            for teacher_seed in (7, 17, 29):
                path = source_path(method, teacher_seed)
                policy = load_policy(path, config, args.device, train[0].spot.dtype)
                with torch.no_grad():
                    losses = torch.stack([rollout(policy, bank)["terminal_loss"] for bank in score_banks])
                    values = [empirical_es(value, config.risk.alpha) for value in losses]
                teacher_loss_cache.append(losses.cpu())
                scores.append(dict(checkpoint=str(path), task_es=values))
        teacher_loss_cache = torch.stack(teacher_loss_cache, dim=1)
        torch.save(dict(losses=teacher_loss_cache, candidates=scores),
                   output / f"{preset}-teacher-losses.pt")
        teacher_scores = [min(item["task_es"][task] for item in scores) for task in range(2)]
        _sync(args.device)
        preset_report = report["presets"].setdefault(preset, dict(config=asdict(config), seeds={}))
        preset_report["teacher"] = dict(candidates=scores, reference_es=teacher_scores,
            selection="Minimum task ES across complete frozen feasible policies, training subset only",
            score_paths_per_task=args.score_paths, rollouts=9*2*args.score_paths,
            seconds=time.perf_counter()-teacher_started)
        write_report()
        for seed in args.seeds:
            original = load_policy(source_path(args.student, seed), config, args.device, train[0].spot.dtype)
            directory = output / preset / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            seed_report = preset_report["seeds"].setdefault(str(seed),
                dict(source_checkpoint=str(source_path(args.student, seed)), results={}))
            if "source" not in seed_report["results"]:
                source_output = directory / "source"
                source_output.mkdir(exist_ok=True)
                seed_report["results"]["source"] = dict(evaluation=evaluate(original, development,
                    source_output, args.device, f"{preset}/{seed}/source"))
                write_report()
            for sampler in args.samplers:
                arm = directory / sampler
                arm.mkdir(exist_ok=True)
                checkpoint = arm / "latest.pt"
                policy, training = train_task_curriculum(original, train, sampler=sampler,
                    teacher_scores=teacher_scores, seed=seed, updates=args.updates,
                    batch_size=args.batch_size, score_paths=args.score_paths,
                    score_every=args.score_every, device=args.device, checkpoint_path=checkpoint,
                    teacher_losses=teacher_loss_cache if sampler == "pooled_ru_regret" else None,
                    resume_from=checkpoint if checkpoint.exists() else None)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                policy.load_state_dict(saved["policy"])
                metrics = evaluate(policy, development, arm, args.device, f"{preset}/{seed}/{sampler}")
                seed_report["results"][sampler] = dict(training=training, evaluation=metrics)
                write_report()
                _report("curriculum_result", preset=preset, seed=seed, sampler=sampler,
                    pooled_es95=metrics["pooled_es95"], A_es95=metrics["A"]["es95"],
                    B_es95=metrics["B"]["es95"], seconds=training["total_seconds"])


if __name__ == "__main__":
    main()
