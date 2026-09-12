"""Qualify official SimBaV2 on saved common banks and terminal ES95.

Set XLA_PYTHON_CLIENT_PREALLOCATE=false before starting. JAX learner and the
CPU Torch ledger exchange NumPy arrays. No donor simulation or logging stack
is used. Checkpoints include donor optimizers, normalizers, replay and RNGs.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import BANK_FIELDS, MarketBank, bank_subset, numpy_ledger
from hedging_gym.gym_env import TensorHedgingEnv
from methods.simbav2 import SOURCE_COMMIT, SimBaV2Hedger, RawLossReplay


def load_bank(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return MarketBank(*(saved[key] for key in BANK_FIELDS), config_from_dict(saved["config"]))


def collect(learner, bank, num_envs, *, random_actions, previous=None):
    """Complete fixed-bank episodes, preserving the donor's observation stream."""
    indices = learner.generator.choice(len(bank.spot), num_envs, replace=False)
    env = TensorHedgingEnv(bank_subset(bank, indices))
    observed = env.reset().numpy().astype(np.float32)
    if previous is None:
        previous = dict(terminal_loss=np.zeros(num_envs), terminated=np.zeros(num_envs, bool))
    for _ in range(bank.config.n_steps):
        prev = dict(reward=RawLossReplay.rewards(previous["terminal_loss"], previous["terminated"],
                    learner.zeta, bank.config.risk.alpha), terminated=previous["terminated"],
                    truncated=np.zeros(num_envs, bool))
        actions = learner.normalized_actions(observed, training=True, previous=prev)
        if random_actions:
            actions = learner.generator.uniform(-1., 1., (num_envs, bank.config.n_assets)).astype(np.float32)
        with torch.no_grad():
            next_observed, _, done, _, info = env.step(torch.from_numpy(learner.holdings(actions)))
        next_observed = next_observed.numpy().astype(np.float32)
        terminated = np.full(num_envs, done, bool)
        raw = info["terminal_loss"].numpy() if done else np.zeros(num_envs)
        learner.replay.add(observed, actions, next_observed, terminated, raw)
        learner.transitions += num_envs
        previous = dict(terminal_loss=raw.copy(), terminated=terminated)
        observed = next_observed
    learner.collections += 1
    learner.set_threshold(learner.zeta)
    return previous


def evaluate(learner, bank, *, seed, deterministic, batch_size=512):
    with learner.evaluation_rng(seed):
        metrics, tape = evaluate_controller(learner.controller(deterministic=deterministic), bank,
                                            device="cpu", batch_size=batch_size, mode_seed=seed)
    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
    error = float(np.max(np.abs(reference["terminal_loss"]-tape["terminal_loss"].numpy())))
    np.testing.assert_allclose(reference["terminal_loss"], tape["terminal_loss"], atol=2e-6, rtol=0)
    return dict(metrics=metrics, cash_error=error), tape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--collections", type=int, default=32)
    parser.add_argument("--warmup-collections", type=int,
                        help="default rounds donor's 5000 warmup transitions up to complete collections")
    parser.add_argument("--updates-per-transition", type=float, default=1.)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--phase-collections", type=int, default=8)
    parser.add_argument("--calibration-paths", type=int, default=4096)
    parser.add_argument("--development-paths", type=int, default=0, help="0 uses the entire saved development bank")
    parser.add_argument("--actor-width", type=int, default=128)
    parser.add_argument("--critic-width", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--jax-platform", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-collections", type=int, help="checkpoint stop within an unchanged total schedule")
    args = parser.parse_args()
    if min(args.envs, args.collections, args.batch_size, args.phase_collections, args.calibration_paths,
           args.actor_width, args.critic_width, args.replay_capacity) < 1 or args.updates_per_transition <= 0:
        parser.error("positive sizes and update ratio required")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_PLATFORMS"] = args.jax_platform
    torch.set_num_threads(args.threads)
    training, development = load_bank(args.run_dir / "train_bank.pt"), load_bank(args.run_dir / "development_bank.pt")
    if args.envs > len(training.spot):
        parser.error("env count exceeds the training bank")
    if training.config != development.config:
        raise ValueError("training and development financial configurations must match")
    calibration = bank_subset(training, slice(0, min(args.calibration_paths, len(training.spot))))
    if args.development_paths:
        development = bank_subset(development, slice(0, args.development_paths))
    transitions_per_collection = args.envs * training.config.n_steps
    if args.warmup_collections is None:
        args.warmup_collections = math.ceil(5000 / transitions_per_collection)
    if not 0 <= args.warmup_collections < args.collections:
        parser.error("warmup must leave at least one training collection")
    expected_updates = int((args.collections-args.warmup_collections)*transitions_per_collection*args.updates_per_transition)
    if expected_updates < 1:
        parser.error("schedule must include an optimizer update")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    # Exclude only output/stop controls; the original schedule is fixed on resume.
    contract = {key: value for key, value in arguments.items()
                if key not in ("output_dir", "resume_from", "stop_after_collections", "donor_path")}
    print(json.dumps(dict(stage="initializing", source_commit=SOURCE_COMMIT, arguments=arguments,
        config=asdict(training.config), transitions=args.collections*transitions_per_collection,
        expected_updates=expected_updates, jax_preallocation=os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"])), flush=True)
    learner = SimBaV2Hedger(training.config, args.donor_path, seed=args.seed, total_updates=expected_updates,
        replay_capacity=args.replay_capacity, actor_width=args.actor_width, critic_width=args.critic_width)
    import jax
    recipe = dict(arguments=arguments, config=asdict(training.config), source_commit=SOURCE_COMMIT,
        donor_config=learner.cfg, devices=[str(device) for device in jax.devices()],
        versions={name: importlib.metadata.version(name) for name in
                  ("jax", "jaxlib", "flax", "optax", "tensorflow-probability", "numpy", "torch", "gymnasium")},
        expected_updates=expected_updates, replay_samples_per_transition=args.updates_per_transition*args.batch_size,
        adaptations=["saved common training bank and CPU ledger with NumPy bridge", "undiscounted terminal global RU reward",
            "training-only sampled-policy threshold recalibration between complete collections",
            "raw-loss replay relabeled at current threshold before original reward normalization",
            "original reward normalizer max-return floor covers relabeled replay terminal rewards",
            "batched complete-episode collection, updates counted per inserted transition",
            "episodic double categorical critic; no behavior cloning; full local checkpoints"],
        initialization_seconds=time.perf_counter()-started)
    (args.output_dir / "recipe.json").write_text(json.dumps(recipe, indent=2)+"\n")
    records, previous, update_credit, previous_seconds = [], None, 0., 0.
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        runner = saved["runner"]
        if runner["contract"] != contract:
            raise ValueError("resume requires the original collection, schedule and bank contract")
        learner.load_state_dict(saved["learner"])
        records, previous = runner["records"], runner["previous"]
        update_credit, previous_seconds = runner["update_credit"], runner["seconds"]
    learning_started = time.perf_counter()
    stop = min(args.collections, args.stop_after_collections or args.collections)
    last_info = {}
    while learner.collections < stop:
        if learner.collections % args.phase_collections == 0:
            with learner.evaluation_rng(91001 + learner.collections):
                _, tape = evaluate_controller(learner.controller(deterministic=False), calibration,
                    device="cpu", batch_size=512, mode_seed=91001+learner.collections)
            learner.set_threshold(float(torch.quantile(tape["terminal_loss"], training.config.risk.alpha)))
        previous = collect(learner, training, args.envs,
                           random_actions=learner.collections < args.warmup_collections, previous=previous)
        collection_started = time.perf_counter()
        if learner.collections > args.warmup_collections:
            update_credit += transitions_per_collection*args.updates_per_transition
            target_updates = int(update_credit + 1e-9)
            for index in range(target_updates):
                last_info = learner.update(args.batch_size)
                if (index+1) % 100 == 0:
                    print(json.dumps(dict(stage="SimBaV2 updates", collection=learner.collections,
                        completed=index+1, total=target_updates, seconds=time.perf_counter()-collection_started)), flush=True)
            update_credit -= target_updates
        elapsed = previous_seconds + time.perf_counter()-learning_started
        record = dict(collections=learner.collections, transitions=learner.transitions, updates=learner.updates,
            zeta=learner.zeta, seconds=elapsed, update_seconds=time.perf_counter()-collection_started,
            reward_scale_max=learner.agent.G_r_max, last_update=last_info,
            eta_seconds=elapsed*(args.collections-learner.collections)/max(1, learner.collections))
        records.append(record)
        print(json.dumps(record), flush=True)
        runner = dict(contract=contract, records=records, previous=previous, update_credit=update_credit, seconds=elapsed)
        learner.save(args.output_dir / "checkpoint.pt", runner=runner)
        (args.output_dir / "progress.json").write_text(json.dumps(records, indent=2)+"\n")
    scores = {}
    for deterministic in (True, False):
        mode = "greedy" if deterministic else "sampled"
        score, tape = evaluate(learner, development, seed=30001, deterministic=deterministic)
        torch.save(tape, args.output_dir / f"development-{mode}.pt")
        scores[mode] = score
    # Test disk restoration of the complete learner, not only policy parameters.
    saved = torch.load(args.output_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    learner.load_state_dict(saved["learner"])
    _, reloaded = evaluate(learner, development, seed=30001, deterministic=True)
    original = torch.load(args.output_dir / "development-greedy.pt", map_location="cpu", weights_only=False)
    torch.testing.assert_close(reloaded["terminal_loss"], original["terminal_loss"], rtol=0, atol=0)
    summary = dict(classification="official SimBaV2 common-task qualification", scores=scores,
        checkpoint_reload="identical deterministic terminal losses", transitions=learner.transitions,
        updates=learner.updates, training_seconds=records[-1]["seconds"],
        total_process_seconds=time.perf_counter()-started, completed_schedule=learner.collections==args.collections)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
