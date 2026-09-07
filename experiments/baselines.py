"""Tiny DEVELOPMENT comparison: existing classical, direct DH and NTB adapters.

Run from a checkout: uv run --frozen python -m experiments.baselines.
The default is a short CPU integration example, not a scientific benchmark.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from hedging_gym.evaluation import evaluate_controller
from hedging_gym.benchmark import benchmark_config
from hedging_gym.finance import BANK_FIELDS, generate_market_bank

from methods.controllers import classical_controller, policy_controller
from methods.training import METHOD_LABELS, _report, train_policy


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--model", default="heston", choices=("gbm", "heston", "bates"))
    parser.add_argument("--methods", nargs="+", choices=("dh", "ntb"), default=["dh", "ntb"])
    parser.add_argument("--train-paths", type=int, default=128)
    parser.add_argument("--eval-paths", type=int, default=128)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--hidden", nargs="+", type=int, default=[32, 32])
    parser.add_argument("--seed", type=int, default=7, help="policy initialization/minibatch seed")
    parser.add_argument("--train-seed", type=int, default=1101)
    parser.add_argument("--eval-seed", type=int, default=2201)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--delta-band", type=float, default=.05,
                        help="predeclared stock-quantity half-width; not tuned on evaluation")
    parser.add_argument("--output-dir", type=Path,
                        help="optional untracked directory for checkpoints, bank and loss tapes")
    return parser


def _build_bank(config, paths, seed, device, stage):
    _report(stage+"_start", paths=paths, seed=seed, device=device,
            expected_decision_states=paths*config.n_steps)
    started = time.perf_counter()
    bank = generate_market_bank(config, paths, seed, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    _report(stage+"_complete", completed=paths, total=paths,
            elapsed_seconds=time.perf_counter()-started, eta_seconds=0)
    return bank, time.perf_counter()-started


def main(argv=None):
    args = _parser().parse_args(argv)
    if min(args.train_paths, args.eval_paths, args.updates, args.batch_size, args.steps, args.threads) < 1:
        raise ValueError("path counts, updates, batch, dates and threads must be positive")
    if args.train_seed == args.eval_seed:
        raise ValueError("training and held-out banks require distinct seeds")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu")
    torch.set_num_threads(args.threads)
    config = benchmark_config(model=args.model, n_steps=args.steps)
    methods = list(dict.fromkeys(args.methods))
    started = time.perf_counter()
    _report("development_start", label="DEVELOPMENT / ADAPTATION; no publication comparison",
            config=asdict(config), device=args.device, workers=args.threads,
            policy_seed=args.seed, train_seed=args.train_seed, eval_seed=args.eval_seed,
            methods=methods, updates_per_method=args.updates, batch_size=args.batch_size,
            train_paths=args.train_paths, eval_paths=args.eval_paths,
            expected_training_episode_rollouts=len(methods)*args.updates*args.batch_size,
            expected_evaluated_policies=4+len(methods), delta_band=args.delta_band)
    train_bank, train_bank_seconds = _build_bank(config, args.train_paths, args.train_seed,
                                                 args.device, "training_bank")
    policies, training = {}, {}
    for method in methods:
        policies[method], training[method] = train_policy(
            method, train_bank, seed=args.seed, updates=args.updates,
            batch_size=args.batch_size, hidden=tuple(args.hidden), device=args.device)
    # This bank is generated only after all policy updates have completed.
    heldout, eval_bank_seconds = _build_bank(config, args.eval_paths, args.eval_seed,
                                             args.device, "heldout_bank")
    controllers = {
        "delta": classical_controller("delta"),
        "delta_band": classical_controller("delta", band=args.delta_band),
        "delta_gamma": classical_controller("delta_gamma"),
        "delta_variance": classical_controller("delta_variance"),
        **{method: policy_controller(policy) for method, policy in policies.items()},
    }
    evaluations, tapes = {}, {}
    for name, controller in controllers.items():
        label = METHOD_LABELS.get(name, "ADAPTATION / model-priced "+name)
        evaluations[name], tapes[name] = evaluate_controller(
            controller, heldout, device=args.device, batch_size=args.batch_size,
            label="DEVELOPMENT / "+label, progress=True,
            zeta=training[name]["zeta"] if name in training else None)
        _report("heldout_result", method=name, eval_seed=args.eval_seed, **evaluations[name])
    summary = dict(label="DEVELOPMENT / ADAPTATION; no publication comparison",
                   config=asdict(config), arguments={**vars(args), "output_dir": str(args.output_dir) if args.output_dir else None},
                   training=training, evaluation=evaluations,
                   training_bank_seconds=train_bank_seconds, heldout_bank_seconds=eval_bank_seconds,
                   total_seconds_before_optional_save=time.perf_counter()-started,
                   scope="One tiny training seed and shared held-out paths; no model selection, HPO or superiority claim",
                   execution="Continuous basic targets; lot and minimum-order adaptations are not implemented")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(config), **{key: getattr(heldout, key).cpu() for key in BANK_FIELDS},
                    "seed": args.eval_seed}, args.output_dir/"heldout-bank.pt")
        torch.save(tapes, args.output_dir/"evaluation-tapes.pt")
        for name, policy in policies.items():
            torch.save({"policy": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                        "metadata": training[name]}, args.output_dir/(name+"-checkpoint.pt"))
        (args.output_dir/"comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    _report("development_complete", total_seconds=time.perf_counter()-started,
            methods=list(controllers), output_dir=str(args.output_dir) if args.output_dir else None,
            claim="Implementation exercise only; tail estimates from this tiny run are not publication evidence")
    return summary


if __name__ == "__main__":
    main()
