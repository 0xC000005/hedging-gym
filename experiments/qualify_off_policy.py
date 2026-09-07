"""CrossQ/TQC pilot or declared-budget run on the common saved Heston banks.

Training uses one critic update per transition after warmup, source network
sizes and entropy tuning. Calibration uses sampled-policy training-bank losses
only; development scores do not select thresholds, hyperparameters or models.
Checkpoints include replay and exact continuation state, not policy weights alone.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from experiments.qualify_sb3 import load_bank
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import bank_subset, numpy_ledger
from methods.sb3 import SB3HedgingVecEnv
from methods.off_policy import (build_off_policy, load_off_policy, off_policy_controller,
                                save_off_policy, set_risk_threshold)


class Progress(BaseCallback):
    def __init__(self, total, episode_steps):
        super().__init__()
        self.total, self.episode_steps = total, episode_steps
        self.started = time.perf_counter()
        self.last_report = self.started

    def _on_training_start(self):
        self.initial_steps = self.num_timesteps

    def _on_step(self):
        now = time.perf_counter()
        if self.num_timesteps % self.episode_steps == 0 or now - self.last_report >= 30:
            elapsed, completed = now-self.started, self.num_timesteps-self.initial_steps
            print(json.dumps(dict(stage="off-policy collection", algorithm=type(self.model).__name__,
                transitions=self.num_timesteps, target=self.total, updates=self.model._n_updates,
                seconds=elapsed, eta_seconds=elapsed*(self.total-self.num_timesteps)/max(1, completed))), flush=True)
            self.last_report = now
        return True


def assess(model, training, development, args, phase):
    started = time.perf_counter()
    calibration = bank_subset(training, slice(0, args.calibration_paths))
    _, sampled = evaluate_controller(off_policy_controller(model, deterministic=False), calibration,
        batch_size=args.eval_batch_size, mode_seed=91001+phase, device=args.device)
    zeta = float(torch.quantile(sampled["terminal_loss"], training.config.risk.alpha))
    record = dict(phase=phase, transitions=model.num_timesteps, updates=model._n_updates,
                  entropy_coefficient=float(model.log_ent_coef.detach().exp().cpu()),
                  zeta=zeta, calibration_paths=len(calibration.spot), scores={})
    for deterministic in (True, False):
        name = "greedy" if deterministic else "sampled"
        metrics, tape = evaluate_controller(off_policy_controller(model, deterministic=deterministic),
            development, batch_size=args.eval_batch_size, mode_seed=30001, zeta=zeta, device=args.device)
        reference = numpy_ledger(development.marks.numpy(), tape["positions"].numpy(),
            development.liability[:,0].numpy(), development.liability[:,-1].numpy(), development.config)
        error = float(np.max(np.abs(reference["terminal_loss"]-tape["terminal_loss"].numpy())))
        np.testing.assert_allclose(reference["terminal_loss"], tape["terminal_loss"], atol=2e-6, rtol=0)
        record["scores"][name] = dict(metrics=metrics, cash_error=error)
        torch.save(tape, args.output_dir / f"phase{phase}-{name}.pt")
    record["assessment_seconds"] = time.perf_counter()-started
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("crossq", "tqc"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--phases", type=int, default=1)
    parser.add_argument("--episodes-per-phase", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--development-paths", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.phases < 1 or args.envs < 1 or args.episodes_per_phase < args.envs
            or args.episodes_per_phase % args.envs or args.calibration_paths < 1
            or args.development_paths < 1):
        parser.error("positive phases/path counts and episodes-per-phase divisible by envs required")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    training = load_bank(args.run_dir / "train_bank.pt")
    development = load_bank(args.run_dir / "development_bank.pt")
    if args.calibration_paths > len(training.spot) or args.development_paths > len(development.spot):
        parser.error("requested assessment subset exceeds the saved bank")
    development = bank_subset(development, slice(0, args.development_paths))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_phase = args.episodes_per_phase * training.config.n_steps
    total = args.phases * steps_per_phase
    recipe = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        config=asdict(training.config), transitions=total, gamma=1., train_freq=1,
        gradient_steps=-1, critic_updates_per_transition_after_warmup=1,
        source="sb3-contrib==2.9.0", network="upstream defaults",
        objective="terminal negative RU loss with training-only zeta; upstream automatic entropy regularization",
        reward_units="raw terminal cash loss; no reward rescaling (saved Heston benchmark spot0=1)",
        entropy_caveat="source auto entropy starts at 1 and may initially dominate small financial losses",
        replay="raw terminal losses relabeled at sampling with current zeta",
        claim_scope="pilot unless a full comparison budget is separately declared")
    print(json.dumps(recipe), flush=True)
    env = SB3HedgingVecEnv(training, args.envs)
    if args.resume:
        model, state = load_off_policy(args.resume, env, device=args.device)
        old = state["recipe"]["arguments"]
        for key in ("algorithm", "run_dir", "seed", "envs", "episodes_per_phase", "buffer_size",
                    "learning_starts", "batch_size", "calibration_paths", "development_paths",
                    "eval_batch_size", "threads", "device"):
            if old[key] != recipe["arguments"][key]:
                parser.error(f"resume must preserve {key}")
        records, completed = state["records"], state["completed_phase"]
        previous_seconds = state["total_seconds"]
        if args.phases <= completed:
            parser.error("resume requires phases greater than the completed phase")
    else:
        model = build_off_policy(args.algorithm, env, seed=args.seed, device=args.device,
            buffer_size=args.buffer_size, learning_starts=args.learning_starts, batch_size=args.batch_size)
        # Initialize the library's continuation state without collecting transitions.
        model.learn(0, reset_num_timesteps=False)
        records, completed, previous_seconds = [], -1, 0.
    recipe.update(learning_rate=model.learning_rate, policy_architecture=model.policy.net_arch,
                  actor_update_delay=getattr(model, "policy_delay", 1),
                  replay_capacity_transitions=model.replay_buffer.buffer_size*args.envs)
    (args.output_dir / "recipe.json").write_text(json.dumps(recipe, indent=2)+"\n")
    for phase in range(completed+1, args.phases+1):
        train_started, updates_before = time.perf_counter(), model._n_updates
        if phase:
            model.learn(steps_per_phase, reset_num_timesteps=False,
                callback=Progress(total, args.envs*training.config.n_steps))
        training_seconds = time.perf_counter()-train_started if phase else 0.
        record = assess(model, training, development, args, phase)
        set_risk_threshold(model, env, record["zeta"])
        record.update(training_seconds=training_seconds,
                      phase_updates=model._n_updates-updates_before,
                      total_seconds=previous_seconds+time.perf_counter()-started)
        records.append(record)
        checkpoint_started = time.perf_counter()
        save_off_policy(model, env, args.output_dir / f"phase{phase}",
            runner_state=dict(recipe=recipe, records=records, completed_phase=phase,
                              total_seconds=record["total_seconds"]))
        record["checkpoint_seconds"] = time.perf_counter()-checkpoint_started
        record["total_seconds"] = previous_seconds+time.perf_counter()-started
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2)+"\n")
        print(json.dumps(record), flush=True)
    verification_started = time.perf_counter()
    reloaded, _ = load_off_policy(args.output_dir / f"phase{args.phases}", env, device=args.device)
    _, tape = evaluate_controller(off_policy_controller(reloaded), development,
        batch_size=args.eval_batch_size, mode_seed=30001, device=args.device)
    original = torch.load(args.output_dir / f"phase{args.phases}-greedy.pt", weights_only=False)
    torch.testing.assert_close(tape["terminal_loss"], original["terminal_loss"], atol=0, rtol=0)
    records[-1]["reload_verification_seconds"] = time.perf_counter()-verification_started
    records[-1]["total_seconds"] = previous_seconds+time.perf_counter()-started
    (args.output_dir / "results.json").write_text(json.dumps(records, indent=2)+"\n")
    print("Final checkpoint reload: identical greedy terminal losses", flush=True)
    env.close()


if __name__ == "__main__":
    main()
