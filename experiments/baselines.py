"""Common baseline training/evaluation, with optional market-only adaptation.

Run from a checkout: uv run --frozen python -m experiments.baselines.
The default is a short CPU integration example, not a scientific benchmark.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from hedging_gym.evaluation import evaluate_controller
from hedging_gym.benchmark import benchmark_config, evaluate_adaptation
from hedging_gym.config import RiskConfig, TimeGrid, config_from_dict
from hedging_gym.finance import BANK_FIELDS, MarketBank, generate_market_bank

from methods.controllers import classical_controller, policy_controller
from methods.training import _report, train_policy
from methods.hybrid import train_hybrid
from methods.planning import RolloutPlanner
from methods.adaptation import train_online_finetune, train_multitask, AdaptationUpdater
from methods.model_free import train_model_free
from methods.alphazero import train_alphazero, alphazero_controller


METHOD_LABELS = {
    "dh": "Deep Hedging", "ntb": "Learned no-transaction bands",
    "hull_rl": "Hull/Rotman QR-D4PG adaptation", "exdrl": "EX-D4PG tail-risk adaptation",
    "finetune_dh": "Online fine-tuned Deep Hedging", "adaptive_dh": "Task-embedding Deep Hedging",
    "alphazero": "Stochastic AlphaZero adaptation", "cem": "Unguided CEM + DH continuation",
    "hpo": "Hybrid policy optimization", "hpo_cem": "HPO-guided CEM",
    "hpo_gradient": "HPO-guided CEM + pathwise refinement",
}
SEARCH_METHODS = {"alphazero", "cem", "hpo_cem", "hpo_gradient"}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--model", default="heston", choices=("gbm", "heston", "bates"))
    parser.add_argument("--methods", nargs="+", choices=("all", *METHOD_LABELS), default=["dh", "ntb"])
    parser.add_argument("--preset", default="basic",
                        choices=("basic", "operational_fixed", "operational_minimum_fee"))
    parser.add_argument("--train-paths", type=int, default=128)
    parser.add_argument("--eval-paths", type=int, default=128)
    parser.add_argument("--train-bank", type=Path, help="reuse a trusted saved bank with the same configuration")
    parser.add_argument("--eval-bank", type=Path, help="reuse a separate saved development/evaluation bank")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=Path, help="periodic complete training snapshots outside Git")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume-from", type=Path, help="resume one selected method on its original training bank")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=TimeGrid().n_steps)
    parser.add_argument("--days-per-year", type=int, choices=(252, 365, 360),
                        default=TimeGrid().days_per_year,
                        help="year clock for trading and contract maturities")
    parser.add_argument("--risk-alpha", type=float,
                        help="terminal expected-shortfall confidence; defaults to RiskConfig")
    parser.add_argument("--hidden", nargs="+", type=int, default=[32, 32])
    parser.add_argument("--seed", type=int, default=7, help="policy initialization/minibatch seed")
    parser.add_argument("--train-seed", type=int, default=1101)
    parser.add_argument("--eval-seed", type=int, default=2201)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--search-simulations", type=int, default=16, help="AlphaZero MCTS simulations per decision")
    parser.add_argument("--grid-points", type=int, default=3, help="AlphaZero holding-grid resolution per asset")
    parser.add_argument("--search-candidates", type=int, default=16, help="CEM candidates per iteration")
    parser.add_argument("--search-scenarios", type=int, default=16, help="CEM conditional paths per candidate")
    parser.add_argument("--search-iterations", type=int, default=2)
    parser.add_argument("--search-batch-size", type=int, default=16)
    parser.add_argument("--adaptation-updates", type=int, default=0,
                        help="optimizer steps per A/B/A stage for fine-tuning and task embeddings; 0 skips")
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


def _load_bank(path, config, seed, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if config_from_dict(saved["config"]) != config or saved["seed"] != seed:
        raise ValueError("saved bank configuration/seed differs from the declared run")
    return MarketBank(*(saved[key] for key in BANK_FIELDS), config)


def main(argv=None):
    args = _parser().parse_args(argv)
    if min(args.train_paths, args.eval_paths, args.updates, args.batch_size, args.steps, args.threads,
           args.search_simulations, args.search_candidates, args.search_scenarios,
           args.search_iterations, args.search_batch_size) < 1 or args.adaptation_updates < 0:
        raise ValueError("path counts, updates, batch, dates and threads must be positive")
    if args.train_seed == args.eval_seed:
        raise ValueError("training and held-out banks require distinct seeds")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu")
    torch.set_num_threads(args.threads)
    config = benchmark_config(model=args.model, name=args.preset,
        time_grid=TimeGrid(n_steps=args.steps, days_per_year=args.days_per_year),
        risk=RiskConfig() if args.risk_alpha is None else RiskConfig(alpha=args.risk_alpha))
    methods = list(METHOD_LABELS) if "all" in args.methods else list(dict.fromkeys(args.methods))
    # Planner variants share a trained continuation, without retraining/cost duplication.
    training_methods = [method for method in methods if method not in {"cem", "hpo_cem", "hpo_gradient"}]
    for required, planners in (("dh", {"cem"}), ("hpo", {"hpo_cem", "hpo_gradient"})):
        if planners.intersection(methods) and required not in training_methods:
            training_methods.append(required)
    if "adaptive_dh" in training_methods and args.updates < 2:
        raise ValueError("adaptive_dh needs at least one update per source market (two total)")
    started = time.perf_counter()
    _report("development_start", label="Baseline comparison",
            config=asdict(config), device=args.device, workers=args.threads,
            policy_seed=args.seed, train_seed=args.train_seed, eval_seed=args.eval_seed,
            methods=methods, updates_per_method=args.updates, batch_size=args.batch_size,
            train_paths=args.train_paths, eval_paths=args.eval_paths,
            expected_training_episode_rollouts=len(training_methods)*args.updates*args.batch_size,
            expected_evaluated_policies=4+len(methods), delta_band=args.delta_band)
    if args.resume_from and len(training_methods) != 1:
        raise ValueError("resume one method at a time using its saved training state")
    if args.train_bank:
        bank_start = time.perf_counter()
        train_bank = _load_bank(args.train_bank, config, args.train_seed, args.device)
        if len(train_bank.spot) != args.train_paths:
            raise ValueError("--train-paths must match the saved bank")
        train_bank_seconds = time.perf_counter() - bank_start
    else:
        train_bank, train_bank_seconds = _build_bank(config, args.train_paths, args.train_seed,
                                                     args.device, "training_bank")
    policies, training, extra_bank_seconds = {}, {}, 0.
    options = dict(seed=args.seed, updates=args.updates, batch_size=args.batch_size,
                   hidden=tuple(args.hidden), device=args.device, learning_rate=args.learning_rate)
    for method in training_methods:
        method_options = dict(options)
        if args.checkpoint_dir:
            method_options.update(checkpoint_path=args.checkpoint_dir / f"{method}-seed{args.seed}" / "latest.pt",
                                  checkpoint_every=args.checkpoint_every)
        if args.resume_from:
            method_options["resume_from"] = args.resume_from
        if method in ("dh", "ntb"):
            pair = train_policy(method, train_bank, **method_options)
        elif method in ("hull_rl", "exdrl"):
            pair = train_model_free(method, train_bank, **method_options)
        elif method == "finetune_dh":
            pair = train_online_finetune(train_bank, **method_options)
        elif method == "adaptive_dh":
            # A nearby source market, not evaluation regime B (.09 variance).
            changes = dict(v0=config.market.v0 * .8)
            if config.market.model in ("heston", "bates"):
                changes["theta"] = config.market.theta * .8
            source_config = replace(config, market=replace(config.market, **changes))
            source_bank, seconds = _build_bank(source_config, args.train_paths,
                args.train_seed + 100000, args.device, "multitask_source_bank")
            extra_bank_seconds += seconds
            pair = train_multitask((train_bank, source_bank), **method_options)
        elif method == "alphazero":
            pair = train_alphazero(train_bank, simulations=args.search_simulations,
                                   grid_points=args.grid_points, **method_options)
        else:
            pair = train_hybrid(train_bank, **method_options)
        policies[method], training[method] = pair
    # This bank is generated only after all policy updates have completed.
    if args.eval_bank:
        bank_start = time.perf_counter()
        heldout = _load_bank(args.eval_bank, config, args.eval_seed, args.device)
        if len(heldout.spot) != args.eval_paths:
            raise ValueError("--eval-paths must match the saved bank")
        eval_bank_seconds = time.perf_counter() - bank_start
    else:
        heldout, eval_bank_seconds = _build_bank(config, args.eval_paths, args.eval_seed,
                                                 args.device, "heldout_bank")
    controllers = {
        "delta": classical_controller("delta"),
        "delta_band": classical_controller("delta", band=args.delta_band),
        "delta_gamma": classical_controller("delta_gamma"),
        "delta_variance": classical_controller("delta_variance"),
    }
    planners = {}
    for method in methods:
        if method == "alphazero":
            controllers[method] = alphazero_controller(policies[method], simulations=args.search_simulations,
                                                      seed=args.seed + 200003, progress=True)
        elif method in {"cem", "hpo_cem", "hpo_gradient"}:
            continuation = "dh" if method == "cem" else "hpo"
            planner = RolloutPlanner(policies[continuation], zeta=training[continuation]["zeta"],
                candidates=args.search_candidates, scenarios=args.search_scenarios,
                iterations=args.search_iterations, seed=args.seed + 200003,
                guided=method != "cem", gradient_steps=3 if method == "hpo_gradient" else 0,
                progress=True)
            controllers[method] = planners[method] = planner
        else:
            controllers[method] = policy_controller(policies[method])
    evaluations, tapes = {}, {}
    for name, controller in controllers.items():
        label = METHOD_LABELS.get(name, "Model-priced "+name)
        evaluations[name], tapes[name] = evaluate_controller(
            controller, heldout, device=args.device,
            batch_size=args.search_batch_size if name in SEARCH_METHODS else args.batch_size,
            label=label, progress=True,
            zeta=training[name]["zeta"] if name in training else None)
        _report("heldout_result", method=name, eval_seed=args.eval_seed, **evaluations[name])
    summary = dict(label="Baseline comparison",
                   config=asdict(config), arguments={key: str(value) if isinstance(value, Path) else value
                                                    for key, value in vars(args).items()},
                   training=training, evaluation=evaluations,
                   planning={name: planner.metadata() for name, planner in planners.items()},
                   training_bank_seconds=train_bank_seconds, extra_training_bank_seconds=extra_bank_seconds,
                   heldout_bank_seconds=eval_bank_seconds,
                   comparison_seconds_before_adaptation=time.perf_counter()-started,
                   scope="Development comparison with one training seed; not a final performance ranking",
                   execution="Continuous methods: bounds/fees supported; lots and minimum orders require discrete action adapters",
                   adaptation={})
    if "alphazero" in controllers:
        summary["planning"]["alphazero"] = dict(controllers["alphazero"].work,
            simulations_per_decision=args.search_simulations, seed=args.seed + 200003)
    # Preserve the exact weights used above, before any chronological adaptation.
    checkpoints = {}
    if args.output_dir:
        for name, policy in policies.items():
            checkpoints[name] = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                                  for key, value in policy.state_dict().items()}
    adaptation_tapes, adapted_checkpoints = {}, {}
    if args.adaptation_updates:
        for name in ("finetune_dh", "adaptive_dh"):
            if name not in policies:
                continue
            updater = AdaptationUpdater(policies[name], metadata=training[name],
                updates=args.adaptation_updates, batch_size=args.batch_size, seed=args.seed + 400003)
            report, raw = evaluate_adaptation(policy_controller(policies[name]),
                train_paths=args.train_paths, eval_paths=args.eval_paths, seed=args.eval_seed + 200000,
                base_config=config, updates_per_stage=1, update=updater,
                device=args.device, batch_size=args.batch_size, progress=True)
            report["optimizer_history"] = updater.history
            summary["adaptation"][name] = report
            adaptation_tapes[name] = raw
            adapted_checkpoints[name] = dict(policy=policies[name].state_dict(),
                                             zeta=float(updater.zeta.detach()), history=updater.history)
    summary["total_seconds_before_optional_save"] = time.perf_counter() - started
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(config), **{key: getattr(heldout, key).cpu() for key in BANK_FIELDS},
                    "seed": args.eval_seed}, args.output_dir/"heldout-bank.pt")
        torch.save(tapes, args.output_dir/"evaluation-tapes.pt")
        if adaptation_tapes:
            torch.save(adaptation_tapes, args.output_dir/"adaptation-tapes.pt")
            torch.save(adapted_checkpoints, args.output_dir/"adapted-checkpoints.pt")
        for name, policy in policies.items():
            torch.save({"policy": checkpoints[name], "metadata": training[name],
                        "checkpoint_stage": "after_initial_training"},
                       args.output_dir/(name+"-checkpoint.pt"))
        (args.output_dir/"comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    _report("development_complete", total_seconds=time.perf_counter()-started,
            methods=list(controllers), output_dir=str(args.output_dir) if args.output_dir else None,
            claim="See the saved configuration, training budget and held-out sample before interpreting results")
    return summary


if __name__ == "__main__":
    main()
