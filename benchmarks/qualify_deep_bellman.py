"""Deep Bellman Hedging qualification: seeds, initial books and value checks.

Trains one Deep Bellman policy per declared seed on a frozen bank, then
evaluates it on fresh paths for the configured book and for declared alternative
initial books without retraining. Optional pathwise Deep Hedging controls train
on the same bank and objective. The critic's initial value is compared with the
realized certainty equivalent of the excess terminal P&L. Development
evaluation is report-only; there is no checkpoint selection.
"""
import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines.deep_bellman_hedging import (
    DeepBellmanLearner,
    DeepBellmanPolicy,
    train_deep_bellman,
)
from hedging_gym.baselines.deep_hedging import train as train_dh
from hedging_gym.baselines.delta import delta_hedge_positions, spot_delta
from hedging_gym.environment.benchmark import benchmark_config
from hedging_gym.environment.config import RiskConfig, config_from_dict
from hedging_gym.environment.finance import (
    BANK_FIELDS,
    MarketBank,
    generate_market_bank,
    initial_state,
    numpy_ledger,
    observation,
)
from hedging_gym.evaluation import empirical_es, evaluate_controller


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="financial configuration JSON; defaults to benchmark_config()")
    parser.add_argument("--objective", choices=("es", "entropy"), help="override config.risk.objective")
    parser.add_argument("--alpha", type=float, help="override config.risk.alpha")
    parser.add_argument("--risk-aversion", type=float, help="override config.risk.risk_aversion")
    parser.add_argument("--utility", help="Deep Bellman utility; defaults to the configured objective")
    parser.add_argument("--position-range", nargs="+", type=float, help="training holdings range per asset")
    parser.add_argument("--initial-positions", action="append", nargs="+", type=float, default=[],
                        help="alternative initial holdings per asset, evaluated without retraining; repeatable")
    parser.add_argument("--compare-dh", action="store_true", help="train pathwise Deep Hedging on the same bank")
    parser.add_argument("--retrain-dh", action="store_true", help="also retrain Deep Hedging per alternative book")
    parser.add_argument("--delta", action="store_true",
                        help="also evaluate the classical delta control; its sensitivities are priced once per bank")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7], help="policy initialization/sampling seeds")
    parser.add_argument("--train-seed", type=int, default=1101)
    parser.add_argument("--eval-seed", type=int, default=2201)
    parser.add_argument("--train-paths", type=int, default=1024)
    parser.add_argument("--eval-paths", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", nargs="+", type=int, default=[32, 32])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--scenarios", type=int, default=1,
                        help="conditional one-day continuations per sampled state; 1 uses the bank's next day")
    parser.add_argument("--steps", type=int, default=1,
                        help="decisions per Bellman target (the paper's T_n); the horizon gives terminal targets")
    parser.add_argument("--aggregate", choices=("oce", "entropic"), default="oce",
                        help="risk aggregation over scenarios: learned OCE shift, or closed-form entropic")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="continue each seed from its latest checkpoint")
    parser.add_argument("--bank-dir", type=Path,
                        help="untracked cache of generated banks and delta targets, keyed by market, calendar and book")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _certainty_equivalent(excess, utility, lam, alpha):
    """Realized monetary utility of excess terminal gains, where a closed form exists."""
    if utility == "identity":
        return float(excess.mean())
    if utility == "entropy":
        return float(-(torch.logsumexp(-lam*excess, 0)-np.log(len(excess)))/lam)
    if utility == "cvar":
        return -empirical_es(-excess, alpha)
    return None


def _books(config, variants):
    books = {"configured": config}
    for index, positions in enumerate(variants, 1):
        books[f"book{index}"] = replace(config, portfolio=replace(config.portfolio, initial_positions=tuple(positions)))
    return books


def _excess_gains(bank, tape):
    state = initial_state(bank)
    wealth = state.cash+(state.positions*bank.marks[:, 0]).sum(-1)-bank.liability[:, 0]
    return -tape["terminal_loss"].double()-wealth.cpu().double(), float(wealth.mean())


def _replay(positions):
    """Deterministic controller replaying precomputed targets; evaluate it in one batch.

    A decision at maturity keeps the preceding holdings: the delta table covers the
    n_steps pre-maturity decisions and settlement liquidates what is left.
    """
    def control(observed, ledger, time_index, config):
        return positions[:, time_index] if time_index < positions.shape[1] else ledger.positions
    control.action_selection = "deterministic"
    return control


def _bank(config, paths, seed, device, bank_dir, stage):
    """Generate a bank, or reuse a cached one with the same market, calendar and book."""
    key = hashlib.sha256(json.dumps({name: asdict(config)[name] for name in ("market", "time_grid", "portfolio")},
                                    sort_keys=True).encode()).hexdigest()[:12]
    path = None if bank_dir is None else bank_dir/f"{stage}-seed{seed}-paths{paths}-{key}.pt"
    if path is not None and path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        return MarketBank(*(saved[field] for field in BANK_FIELDS), config), saved, path
    bank = generate_market_bank(config, paths, seed, device=device)
    saved = {field: getattr(bank, field).cpu() for field in BANK_FIELDS}
    if path is not None:
        bank_dir.mkdir(parents=True, exist_ok=True)
        torch.save(saved, path)
    return bank, saved, path


def _bank_deltas(bank):
    """Liability deltas [paths, n_steps] for every decision state of a bank.

    Delta targets depend on the market state only, so one pricing pass serves
    every seed and every initial book. States are priced date by date with the
    variance sorted within each date: the Heston quadrature grid of a chunk is
    set by its lowest-variance, shortest-maturity state, so grouping the many
    zero-variance states keeps the other chunks cheap.
    """
    config = bank.config
    spot, variance = bank.spot[:, :-1].T, bank.variance[:, :-1].T
    times = torch.arange(config.n_steps, device=spot.device)[:, None].expand_as(spot)
    rows = torch.arange(config.n_steps, device=spot.device)[:, None]
    order = variance.argsort(dim=1)
    flat = spot_delta(spot[rows, order].flatten(), variance[rows, order].flatten(), times.flatten(),
                      config, chunk_size=256)
    deltas = torch.empty_like(spot)
    deltas[rows, order] = flat.reshape(spot.shape)
    return deltas.T


def main(argv=None):
    args = _parser().parse_args(argv)
    if min(args.train_paths, args.eval_paths, args.updates, args.batch_size, args.critic_steps,
           args.checkpoint_every, args.threads, len(args.seeds)) < 1:
        raise ValueError("path counts, updates, batch, steps and threads must be positive")
    if args.train_seed == args.eval_seed:
        raise ValueError("training and held-out banks require distinct seeds")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu")
    torch.set_num_threads(args.threads)
    config = benchmark_config() if args.config is None else config_from_dict(json.loads(args.config.read_text()))
    if args.objective or args.alpha is not None or args.risk_aversion is not None:
        risk = config.risk
        config = replace(config, risk=RiskConfig(
            alpha=risk.alpha if args.alpha is None else args.alpha,
            objective=args.objective or risk.objective,
            risk_aversion=risk.risk_aversion if args.risk_aversion is None else args.risk_aversion))
    books = _books(config, args.initial_positions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_bank, _, _ = _bank(config, args.train_paths, args.train_seed, args.device, args.bank_dir, "train")
    eval_bank, eval_saved, eval_path = _bank(config, args.eval_paths, args.eval_seed, args.device, args.bank_dir, "eval")
    bank_seconds = time.perf_counter()-started
    deltas = None
    if args.delta:
        if "deltas" in eval_saved:
            deltas = eval_saved["deltas"].to(args.device)
        else:
            deltas = _bank_deltas(eval_bank)
            if eval_path is not None:
                torch.save(dict(eval_saved, deltas=deltas.cpu()), eval_path)
    delta_seconds = time.perf_counter()-started-bank_seconds
    recipe = dict(updates=args.updates, batch_size=args.batch_size, hidden=tuple(args.hidden),
                  learning_rate=args.learning_rate, device=args.device, progress=True,
                  checkpoint_every=args.checkpoint_every)
    report = dict(label="Deep Bellman Hedging qualification", config=asdict(config),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  git_revision=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                              cwd=Path(__file__).resolve().parent).stdout.strip(),
                  lockfile_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/"uv.lock").read_bytes()).hexdigest(),
                  books={name: book.portfolio.initial_positions for name, book in books.items()},
                  bank_seconds=bank_seconds, delta_seconds=delta_seconds, runs={})
    for seed in args.seeds:
        run_dir = args.output_dir/f"dbh-seed{seed}"
        checkpoint = run_dir/"latest.pt"
        policy, metadata = train_deep_bellman(train_bank, seed=seed, utility=args.utility,
            critic_learning_rate=args.critic_learning_rate, critic_steps=args.critic_steps,
            scenarios=args.scenarios, steps=args.steps, aggregate=args.aggregate,
            position_range=args.position_range, checkpoint_path=checkpoint,
            resume_from=checkpoint if args.resume and checkpoint.exists() else None, **recipe)
        torch.save(dict(policy=policy.state_dict(), config=asdict(config), metadata=metadata), run_dir/"policy.pt")
        options = metadata["options"]
        learner = DeepBellmanLearner(config, hidden=tuple(options["hidden"]), position_range=options["position_range"],
            utility=options["utility"], risk_aversion=options["risk_aversion"], learning_rate=options["learning_rate"],
            critic_learning_rate=options["critic_learning_rate"], device=args.device, dtype=train_bank.spot.dtype,
            scenarios=options["scenarios"], steps=options["steps"], aggregate=options["aggregate"])
        learner.load_state_dict(torch.load(checkpoint, weights_only=False, map_location=args.device)["learner"])
        controls = {}
        if args.compare_dh:
            controls["dh"] = train_dh(train_bank, seed=seed, checkpoint_path=run_dir/"dh-latest.pt", **recipe)
        evaluations, tapes = {}, {}
        for name, book in books.items():
            heldout = replace(eval_bank, config=book)
            entries = {"dbh": (policy_controller(policy), None)}
            if deltas is not None:
                entries["delta"] = (_replay(delta_hedge_positions(heldout, deltas=deltas)), None)
            if "dh" in controls:
                entries["dh"] = (policy_controller(controls["dh"][0]), controls["dh"][1]["zeta"])
            if args.retrain_dh and name != "configured":
                retrained = train_dh(replace(train_bank, config=book), seed=seed,
                                     checkpoint_path=run_dir/f"dh-{name}-latest.pt", **recipe)
                entries["dh_retrained"] = (policy_controller(retrained[0]), retrained[1]["zeta"])
            evaluations[name] = {}
            for method, (controller, zeta) in entries.items():
                evaluations[name][method], tapes[f"{name}/{method}"] = evaluate_controller(
                    controller, heldout, device=args.device, zeta=zeta, progress=True,
                    batch_size=len(heldout.spot) if method == "delta" else 1024,
                    label=f"{method} on {name}, seed {seed}")
            excess, wealth = _excess_gains(heldout, tapes[f"{name}/dbh"])
            with torch.no_grad():
                value = float(learner.value(observation(heldout, 0, initial_state(heldout))).mean())
            evaluations[name]["value_check"] = dict(
                initial_wealth=wealth, critic_value_initial=value,
                realized_certainty_equivalent=_certainty_equivalent(
                    excess, options["utility"], options["risk_aversion"], config.risk.alpha),
                excess_gain_mean=float(excess.mean()))
        # Reload from disk and reconcile the configured book with the independent cash recursion.
        saved = torch.load(run_dir/"policy.pt", weights_only=False, map_location="cpu")
        restored = DeepBellmanPolicy(config, tuple(options["hidden"]), position_range=options["position_range"])
        restored = restored.to(device=args.device, dtype=eval_bank.spot.dtype).eval()
        restored.load_state_dict(saved["policy"])
        _, reloaded = evaluate_controller(policy_controller(restored), eval_bank, device=args.device,
                                          batch_size=1024, label=f"reloaded dbh, seed {seed}")
        tape = tapes["configured/dbh"]
        ledger = numpy_ledger(eval_bank.marks.cpu().numpy(), tape["positions"].numpy(),
            eval_bank.liability[:, 0].cpu().numpy(), eval_bank.liability[:, -1].cpu().numpy(), config)
        verification = dict(
            maximum_reload_loss_difference=float((reloaded["terminal_loss"]-tape["terminal_loss"]).abs().max()),
            maximum_reload_action_difference=float((reloaded["positions"]-tape["positions"]).abs().max()),
            maximum_independent_ledger_difference=float(np.abs(ledger["terminal_loss"]-tape["terminal_loss"].numpy()).max()),
            constraint_violations=evaluations["configured"]["dbh"]["constraint_violations"])
        torch.save(tapes, run_dir/"tapes.pt")
        report["runs"][seed] = dict(training=metadata, controls={name: pair[1] for name, pair in controls.items()},
                                    evaluations=evaluations, verification=verification)
    report["total_seconds"] = time.perf_counter()-started
    (args.output_dir/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({seed: {name: {method: metrics.get("objective_value", metrics)
                                    for method, metrics in books_.items()}
                             for name, books_ in run["evaluations"].items()}
                      for seed, run in report["runs"].items()}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
