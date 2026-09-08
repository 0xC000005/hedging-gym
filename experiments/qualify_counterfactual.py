"""Bounded development comparison of all-mode versus sampled categorical updates.

Uses immutable stage snapshots as inputs and writes only to a new output folder.
Training-bank stochastic losses calibrate one threshold per source policy, shared
by both update variants and frozen throughout. Development ES is not final-test
evidence. Runtime, branch work and threshold preparation are reported separately.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from hedging_gym.evaluation import evaluate_controller
from methods.checkpoints import load_checkpoint
from methods.controllers import policy_controller
from methods.counterfactual import train_counterfactual
from methods.hybrid import HybridPolicy
from methods.training import _report
from experiments.qualify_policies import load_bank


def sampled_controller(policy):
    """Causal sampled deployment using the evaluator's local RNG contract."""
    def control(observed, ledger, time_index, config):
        policy.check_config(config)
        return policy(observed, ledger.positions, config.execution.holding_lower,
                      config.execution.holding_upper, deterministic=False).target_holdings
    control.action_selection = "sampled categorical modes; frozen conditional sizes"
    return control


def _load_policy(saved, config, device, dtype):
    policy = HybridPolicy(config, hidden=saved["options"]["hidden"]).to(device=device, dtype=dtype)
    policy.load_state_dict(saved["policy"])
    return policy.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", choices=("basic", "operational_fixed"), default="operational_fixed")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--evaluation-batch-size", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source, output = args.source_dir.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("output must be a new directory outside the source evidence")
    if output.exists():
        raise FileExistsError("output directory already exists; choose a new pilot directory")
    train = load_bank(source / "train_bank.pt", args.preset)
    development = load_bank(source / "development_bank.pt", args.preset)
    if asdict(train.config) != asdict(development.config):
        raise ValueError("training and development configurations differ")
    policy_root = source / "policies"
    source_step = 3000
    if args.preset == "operational_fixed":
        policy_root = policy_root / args.preset
        source_step = 1000
    checkpoints = {seed: policy_root / f"hpo-seed{seed}" / f"step-{source_step}.pt" for seed in args.seeds}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    report = dict(scope="Development mechanism pilot; no novelty or superiority claim",
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        config=asdict(train.config), primary_deployment="sampled", secondary_deployment="greedy",
        work_contract="Matched decision ledger steps by per-date batch rounding; elapsed time and liquidation work reported",
        seeds={})
    def write_report():
        (output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    write_report()
    for seed in args.seeds:
        saved = load_checkpoint(checkpoints[seed], method="hpo", config=train.config)
        if saved["seed"] != seed or saved["step"] != source_step:
            raise ValueError("source checkpoint seed/stage mismatch")
        teacher = _load_policy(saved, train.config, args.device, train.spot.dtype)
        directory = output / f"seed-{seed}"
        directory.mkdir()
        _report("counterfactual_source", source_checkpoint=str(checkpoints[seed]), seed=seed,
                device=args.device, config=asdict(train.config), train_paths=len(train.spot),
                development_paths=len(development.spot), updates=args.updates, batch_size=args.batch_size)
        started = time.perf_counter()
        calibration_metrics, calibration_tape = evaluate_controller(sampled_controller(teacher), train,
            device=args.device, batch_size=args.evaluation_batch_size, mode_seed=seed+310003,
            label=f"source-{seed}/training-threshold", progress=True)
        zeta = float(torch.quantile(calibration_tape["terminal_loss"], train.config.risk.alpha))
        torch.save(calibration_tape["terminal_loss"], directory / "threshold-training-losses.pt")
        seed_report = dict(source_checkpoint=str(checkpoints[seed]), fixed_zeta=zeta,
            source_training_metadata={key: saved[key] for key in ("seed", "step", "options", "elapsed_seconds")},
            threshold_calibration=dict(evaluation=calibration_metrics, seconds=time.perf_counter()-started,
                                       scope="Once on all source training paths; shared by both arms"),
            results={})
        report["seeds"][str(seed)] = seed_report
        for algorithm in ("source", "all_mode", "sampled"):
            if algorithm == "source":
                policy, training = teacher, None
            else:
                checkpoint = directory / algorithm / "latest.pt"
                _, training = train_counterfactual(teacher, train, algorithm=algorithm, zeta=zeta,
                    seed=seed, updates=args.updates, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, device=args.device, checkpoint_path=checkpoint)
                policy = _load_policy(load_checkpoint(checkpoint, method=algorithm, config=train.config),
                                      train.config, args.device, train.spot.dtype)
            result = dict(training=training, evaluation={})
            for deployment, controller in (("sampled", sampled_controller(policy)),
                                           ("greedy", policy_controller(policy))):
                metrics, tape = evaluate_controller(controller, development, device=args.device,
                    batch_size=args.evaluation_batch_size, mode_seed=seed+410003,
                    label=f"{algorithm}-{seed}/{deployment}/development", zeta=zeta, progress=True)
                torch.save(tape, directory / f"{algorithm}-{deployment}-tape.pt")
                result["evaluation"][deployment] = metrics
            seed_report["results"][algorithm] = result
            write_report()
            _report("counterfactual_result", algorithm=algorithm, seed=seed,
                    sampled_es95=result["evaluation"]["sampled"]["es95"],
                    greedy_es95=result["evaluation"]["greedy"]["es95"])


if __name__ == "__main__":
    main()
