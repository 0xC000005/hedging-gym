"""One short source-MCTS self-play/refit/save/reload cycle on the common Gym.

This qualifies a port, not competitive training or a published-table reproduction.
Run artifacts and the externally fetched donor stay outside this repository.
"""
import argparse
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from hedging_gym import benchmark_config, finance
from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from methods.source_alphazero import SourceHedgingGame, load_source, source_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    market_input = parser.add_mutually_exclusive_group()
    market_input.add_argument("--market", choices=("heston", "gbm"),
                              help="common benchmark preset; defaults to Heston")
    market_input.add_argument("--config", type=Path,
                              help="complete financial configuration JSON; replaces all benchmark defaults")
    parser.add_argument("--objective", choices=("mse", "es"),
                        help="override config.risk.objective; preset runs default to MSE")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--validation-paths", type=int, default=8)
    parser.add_argument("--calibration-paths", type=int, default=64)
    parser.add_argument("--eval-paths", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--grid-points", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    config = (config_from_dict(json.loads(args.config.read_text())) if args.config is not None
              else benchmark_config(model=args.market or "heston"))
    objective = args.objective or (config.risk.objective if args.config is not None else "mse")
    config = replace(config, risk=replace(config.risk, objective=objective))
    args.objective = objective
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.output/"training.log"),
                                                     logging.StreamHandler(sys.stdout)])
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    started = time.monotonic()
    source_commit = subprocess.check_output(["git", "-C", str(args.donor), "rev-parse", "HEAD"], text=True).strip()
    source_diff = subprocess.check_output(["git", "-C", str(args.donor), "diff"], text=True)
    (args.output/"source.diff").write_text(source_diff)
    record = dict(command=sys.argv, source_commit=source_commit, config=asdict(config),
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        classification="short_common_Gym_source_loop_qualification_not_competitive_training",
        objective=args.objective, primary_metric="mse" if args.objective == "mse" else "expected_shortfall",
        device="cpu", search_pricer="QuantLib analytic; same model and contracts",
        source_url="https://github.com/plan64/minimalHedger_AlphaZero")
    (args.output/"run.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(dict(event="start", **record)), flush=True)
    Trainer, Wrapper = load_source(args.donor)
    scale = config.market.spot0*np.sqrt(config.market.v0*config.time_grid.horizon)
    game = SourceHedgingGame(config, seed=args.seed+101, zeta=0., scale=scale,
                            objective=args.objective, grid_points=args.grid_points, output=args.output)
    nn_args = dict(lr=.001, dropout=.3, epochs=args.epochs,
                   batch_size=args.batch_size, num_channels=args.width)
    network = Wrapper(game, nn_args)
    network.save_checkpoint(str(args.output), "initial.pt")
    if args.objective == "es":
        print(f"CALIBRATION: {args.calibration_paths} fresh complete paths, {config.n_decisions} decisions", flush=True)
        calibration = finance.generate_market_bank(config, args.calibration_paths, args.seed+201)
        _, initial_tape = evaluate_controller(source_controller(network, game), calibration, progress=True)
        game.zeta = float(torch.quantile(initial_tape["terminal_loss"].double(), config.risk.alpha))
        torch.save(initial_tape, args.output/"calibration.pt")
    zeta = game.zeta if args.objective == "es" else None
    settings = dict(learningCycles=args.cycles, episodes=args.episodes, temp=100,
        nnUpdateThreshold=1., numMCTSSims=args.simulations, validationCycles=args.validation_paths,
        nnWeight=1., shortenHistoryAfterIter=5, nnArgs=nn_args,
        saveCheckpointsFolder=str(args.output.resolve())+"/")
    record.update(source_settings=settings, zeta=zeta, reward_scale=scale,
                  actions=game.targets.tolist(), feature_dim=game.feature_dim)
    (args.output/"run.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"TRAIN: {args.cycles} cycle(s), {args.episodes} episodes/cycle, {config.n_decisions} decisions, "
          f"{args.simulations} simulations/decision, {len(game.targets)} actions; "
          f"objective={args.objective}, zeta={zeta}", flush=True)
    trainer = Trainer(game, network, settings)
    trainer.learn()
    network.save_checkpoint(str(args.output), "selected.pt")
    record.update(fits=network.fit_records, search=dict(game.search_stats), transitions=game.transitions,
                  training_elapsed_seconds=time.monotonic()-started)
    print(json.dumps(dict(event="training_complete", **{k: record[k] for k in
                      ("fits", "search", "transitions", "training_elapsed_seconds")})), flush=True)
    assert network.fit_records and all(x["weights_changed"] for x in network.fit_records)
    restored = Wrapper(game, nn_args)
    restored.load_checkpoint(str(args.output), "selected.pt")
    probe = torch.tensor(np.stack([game.observe(game.initial)]*2), dtype=torch.float32)
    for original, loaded in zip(network.predict_batch(probe), restored.predict_batch(probe)):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
    print(f"EVALUATION: {args.eval_paths} new shared paths; initial, candidate, selected", flush=True)
    evaluation = finance.generate_market_bank(config, args.eval_paths, args.seed+301)
    results = {}
    for label, filename in (("initial", "initial.pt"), ("candidate", f"candidate-{network.fit_count}.pt"),
                             ("selected", "selected.pt")):
        restored.load_checkpoint(str(args.output), filename)
        metrics, tape = evaluate_controller(source_controller(restored, game), evaluation,
            progress=True, zeta=zeta, label=label)
        reference = finance.numpy_ledger(evaluation.marks.double().numpy(), tape["positions"].double().numpy(),
            finance.initial_state(evaluation).cash.double().numpy(), evaluation.liability[:, -1].double().numpy(), config)
        error = float(np.max(np.abs(reference["terminal_loss"]-tape["terminal_loss"].numpy())))
        assert error < 2e-6, error
        metrics["cash_reconstruction_max_error"] = error
        results[label] = metrics
        torch.save(tape, args.output/f"evaluation-{label}.pt")
        print(json.dumps(dict(event="evaluated", method=label, objective=args.objective,
            mse=metrics["mse"], es95=metrics["es95"], cash_error=error)), flush=True)
    record.update(results=results, checkpoint_roundtrip="exact", total_seconds=time.monotonic()-started)
    (args.output/"results.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"DONE: {record['total_seconds']:.1f}s; source loop executed, not a superiority claim", flush=True)


if __name__ == "__main__":
    main()
