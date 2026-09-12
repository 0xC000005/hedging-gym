"""Checkpointed AlphaZero training and development evaluation on saved banks.

This uses independent search chance draws, not the future development paths.
Run from a checkout with ``python -m benchmarks.qualify_alphazero --help``.
Development results diagnose the implementation; they are not final test claims.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from hedging_gym.baselines._shared.training import _report
from hedging_gym.baselines.alphazero import (
    AlphaZeroPolicy,
    alphazero_controller,
    completed_rollout_action_diagnostic,
    train_alphazero,
)
from hedging_gym.environment.config import config_from_dict
from hedging_gym.environment.finance import MarketBank, bank_subset
from hedging_gym.evaluation import evaluate_controller


def _bank(path):
    saved = torch.load(path, weights_only=False)
    return MarketBank(**{key: saved[key] for key in ("spot", "variance", "marks", "liability")},
                      config=config_from_dict(saved["config"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, help="Read existing banks separately from new output artifacts")
    parser.add_argument("--name", default="seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--replay-batches", type=int, default=8)
    parser.add_argument("--calibration-paths", type=int, default=1024)
    parser.add_argument("--reanalysis-states", type=int, default=128)
    parser.add_argument("--reanalysis-samples", type=int, default=32)
    parser.add_argument("--value-gradient-steps", type=int, default=128)
    parser.add_argument("--diagnostic-samples", type=int, default=128)
    parser.add_argument("--grid-points", type=int, default=3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--eval-paths", type=int, default=8192)
    parser.add_argument("--search-eval-paths", type=int, default=256)
    parser.add_argument("--eval-simulations", type=int, nargs="*", default=[])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    directory = args.run_root / "alphazero" / args.name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "latest.pt"
    started = time.perf_counter()
    bank_root = args.bank_root or args.run_root
    train = _bank(bank_root / "train_bank.pt")
    if args.evaluate_only:
        saved = torch.load(args.resume or checkpoint, weights_only=False)
        if saved["options"].get("value_target_version") != "raw_loss_greedy_reanalysis_v3":
            raise ValueError("This repaired evaluator needs a v3 checkpoint; use the matching source for legacy policies")
        policy = AlphaZeroPolicy(train.config, hidden=saved["options"]["hidden"],
            targets=saved["options"]["action_targets"]).to(device=args.device, dtype=train.spot.dtype)
        policy.load_state_dict(saved["policy"])
        metadata = dict(zeta=float(policy.zeta), completed=saved["step"])
    else:
        policy, metadata = train_alphazero(train, seed=args.seed, updates=args.updates,
            batch_size=args.batch_size, hidden=tuple(args.hidden), device=args.device,
            simulations=args.simulations, grid_points=args.grid_points,
            gradient_steps=args.gradient_steps, replay_batches=args.replay_batches,
            calibration_paths=args.calibration_paths, reanalysis_states=args.reanalysis_states,
            reanalysis_samples=args.reanalysis_samples, value_gradient_steps=args.value_gradient_steps,
            checkpoint_path=checkpoint, checkpoint_every=4, resume_from=args.resume)
        (directory / "training.json").write_text(json.dumps(metadata, indent=2)+"\n")
    development = _bank(bank_root / "development_bank.pt")
    torch.save(dict(policy=policy.state_dict(), config=asdict(train.config),
                    hidden=list(args.hidden) if not args.evaluate_only else saved["options"]["hidden"],
                    action_targets=policy.targets.cpu().tolist(),
                    step=metadata.get("completed", args.updates)), directory / "evaluated-policy.pt")
    _report("az_action_diagnostic_start", samples=args.diagnostic_samples,
            actions=len(policy.targets), dates=train.config.n_steps,
            expected_maximum_transition_samples=args.diagnostic_samples*(len(policy.targets)+1)*train.config.n_steps)
    diagnostic, diagnostic_tape = completed_rollout_action_diagnostic(policy, train.config,
        samples=args.diagnostic_samples, seed=args.seed+600003)
    torch.save(diagnostic_tape, directory / "completed-rollout-actions.pt")
    (directory / "completed-rollout-actions.json").write_text(json.dumps(diagnostic, indent=2)+"\n")
    _report("az_action_diagnostic", **diagnostic)
    search_simulations = args.eval_simulations
    if diagnostic["action_ordering_screen_failed"] and search_simulations:
        _report("az_search_not_qualified", reason="completed_continuation_action_ordering_screen_failed",
                requested_simulations=search_simulations)
        search_simulations = []
    results = {}
    for simulations in (0, *search_simulations):
        paths = args.eval_paths if simulations == 0 else args.search_eval_paths
        bank = bank_subset(development, slice(0, paths))
        controller = alphazero_controller(policy, simulations=simulations,
            seed=args.seed+400003, progress=simulations > 0)
        result, tape = evaluate_controller(controller, bank, device=args.device,
            batch_size=1024 if simulations == 0 else args.batch_size,
            zeta=metadata["zeta"], label=f"AlphaZero simulations={simulations}", progress=True)
        result.update(paths=len(bank.spot), work=controller.work, simulations=simulations,
                      checkpoint_step=metadata.get("completed", args.updates))
        results[str(simulations)] = result
        torch.save(tape, directory / f"development-sim{simulations}.pt")
        (directory / "development.json").write_text(json.dumps(results, indent=2)+"\n")
        _report("alphazero_development", **result)
    _report("qualification_complete", directory=str(directory),
            elapsed_seconds=time.perf_counter()-started, scope="development_not_final_test")


if __name__ == "__main__":
    main()
