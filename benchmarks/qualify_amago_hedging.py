"""Literal native-AMAGO finance interface qualification, not an ES ranking.

Uses the author agent/Transformer/replay, observed Heston parameters, common
cash ledger and a source-training-only fixed RU threshold. No final test bank
is read. Runtime and full native optimizer checkpoints are retained externally.
"""

import argparse
import json
import random
import time
from pathlib import Path

import amago
import numpy as np
import torch
from amago import cli_utils
from amago.envs import AMAGOEnv
from amago.loading import DiskTrajDataset
from amago.nets.transformer import VanillaAttention

from benchmarks.qualify_adaptation import load_bank
from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines.amago import (
    AmagoController,
    ExplicitEpsilonGreedy,
    MemoryHedgingTask,
)
from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.environment.benchmark import adaptation_configs
from hedging_gym.environment.finance import generate_market_bank
from hedging_gym.evaluation import evaluate_controller


class ProgressExperiment(amago.Experiment):
    """Print actual optimizer steps at each author-defined epoch boundary."""

    def save_checkpoint(self):
        super().save_checkpoint()
        print(json.dumps(dict(event="native_epoch_complete", epoch=self.epoch,
            optimizer_updates=self.grad_update_counter, device=str(self.DEVICE))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--traj-encoder", choices=("transformer", "ff"), default="transformer")
    parser.add_argument("--agent-type", choices=("agent", "multitask"), default="agent",
        help="Official AMAGO Agent or AMAGO-2 MultiTaskAgent; no custom training update")
    parser.add_argument("--load-weights", type=Path)
    parser.add_argument("--context-review", action="store_true")
    parser.add_argument("--context-paths", type=int, default=2048)
    parser.add_argument("--vectorized-actors", type=int, default=0,
        help="Use official already_vectorized collection over the shared TensorHedgingEnv")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    banks = (load_bank(args.source / "train_bank.pt"),
             load_bank(args.source / "adaptation" / "source-nearby.pt"))
    checkpoint = args.source / "policies" / f"dh-seed{args.seed}" / "latest.pt"
    source_policy = torch.load(checkpoint, map_location="cpu", weights_only=False)
    threshold = float(source_policy["zeta"])
    config = {}
    encoder_options = dict(attention_type=VanillaAttention) if args.traj_encoder == "transformer" else {}
    trajectory = cli_utils.switch_traj_encoder(config, arch=args.traj_encoder,
        memory_size=128, layers=3, **encoder_options)
    timestep = cli_utils.switch_tstep_encoder(config, arch="ff", n_layers=1,
        d_hidden=128, d_output=64, normalize_inputs=False)
    cli_utils.switch_exploration(config, strategy="egreedy", eps_start=.2,
        eps_end=.05, steps_anneal=100_000, randomize_eps=True)
    agent = cli_utils.switch_agent(config, args.agent_type, gamma=1., use_multigamma=False,
        reward_multiplier=100., tau=.004)
    (args.output / "native-settings.json").write_text(json.dumps(dict(
        source_commit="54c25ab6da9371614c47352f569a56c0fe938d3b",
        agent_class=f"{agent.__module__}.{agent.__name__}",
        source_paper="https://arxiv.org/abs/2411.11188" if args.agent_type == "multitask"
            else "https://arxiv.org/abs/2310.09971",
        overrides=config, seed=args.seed,
        objective="fixed source-training RU threshold; no joint ES optimization",
        epochs=args.epochs, batches_per_epoch=args.batches,
        actors=args.vectorized_actors or 8), indent=2, default=str)+"\n")
    cli_utils.use_config(config)
    seed_generator = np.random.default_rng(args.seed+900000)
    training_tasks = []

    def make_env(*, training=False):
        task = MemoryHedgingTask(banks, threshold=threshold, episodes=3,
            seed=int(seed_generator.integers(2**31)), num_envs=args.vectorized_actors or 1)
        if training:
            training_tasks.append(task)
        return AMAGOEnv(task,
            env_name="observed_heston_fixed_RU", batched_envs=args.vectorized_actors or 1)

    length = banks[0].config.n_steps * 3
    dataset = DiskTrajDataset(dset_root=str(args.output), dset_name="finance",
        dset_max_size=12_500)
    experiment = ProgressExperiment(run_name="finance", ckpt_base_dir=str(args.output),
        max_seq_len=length, traj_save_len=length, dataset=dataset,
        make_train_env=lambda: make_env(training=True), make_val_env=make_env,
        agent_type=agent, exploration_wrapper_type=ExplicitEpsilonGreedy,
        tstep_encoder_type=timestep, traj_encoder_type=trajectory,
        dloader_workers=0, log_to_wandb=False, epochs=args.epochs,
        parallel_actors=args.vectorized_actors or 8, batch_size=8, train_timesteps_per_epoch=length,
        train_batches_per_epoch=args.batches, val_interval=1,
        val_timesteps_per_epoch=length, ckpt_interval=1,
        env_mode="already_vectorized" if args.vectorized_actors else "sync")
    started = time.perf_counter()
    print(json.dumps(dict(event="finance_start", seed=args.seed,
        agent_type=args.agent_type,
        trajectory_encoder=args.traj_encoder,
        expected_updates=args.epochs*args.batches, threshold=threshold,
        expected_training_transitions=args.epochs*length*(args.vectorized_actors or 8),
        claim="Official-AMAGO fixed-RU finance qualification, not optimal ES or adaptation")), flush=True)
    experiment.start()
    if args.load_weights:
        experiment.load_checkpoint_from_path(str(args.load_weights), is_accelerate_state=False)
    else:
        experiment.learn()
    elapsed = time.perf_counter()-started
    completed_losses = [loss for task in training_tasks for loss in task.terminal_losses]
    completed_losses = np.concatenate(completed_losses) if completed_losses else np.empty(0)
    np.save(args.output / "collected-training-losses.npy", completed_losses)
    development = load_bank(args.source / "development_bank.pt")
    experiment.policy.eval()
    metrics, tape = evaluate_controller(AmagoController(experiment.policy), development,
        device=str(experiment.DEVICE), batch_size=128, zeta=threshold,
        label="AMAGO/fixed-RU/development", progress=True)
    torch.save(tape, args.output / "development-tape.pt")
    report = dict(classification="Literal native-AMAGO fixed-RU finance qualification",
        claim_limits=["No jointly optimized risk threshold",
                      "Base development result resets memory; optional context report is separate",
                      "Not a paper reproduction or matched-compute ranking"],
        source_commit="54c25ab6da9371614c47352f569a56c0fe938d3b",
        source_changes=["Gymnasium1.2 explicit forwarding of two exploration properties",
            "Terminal common RU reward", "gamma1 without auxiliary discount objectives",
            "Official VanillaAttention when Transformer selected",
            "Common Heston banks and three-book training sequences"],
        seed=args.seed, trajectory_encoder=args.traj_encoder, agent_type=args.agent_type,
        threshold=threshold, threshold_source=str(checkpoint),
        requested_updates=args.epochs*args.batches, actual_updates=experiment.grad_update_counter,
        loaded_weights=str(args.load_weights) if args.load_weights else None,
        parallel_actors=args.vectorized_actors or 8,
        actual_completed_books=sum(task.completed_books for task in training_tasks),
        actual_training_threshold_exceedances=int((completed_losses > threshold).sum()),
        device=str(experiment.DEVICE), training_seconds=elapsed, metrics=metrics)
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(event="finance_complete", **report)), flush=True)
    if args.context_review:
        direct = DirectDHPolicy(banks[0].config,
            hidden=source_policy["options"]["hidden"]).to(experiment.DEVICE)
        direct.load_state_dict(source_policy["policy"])
        context_report = dict(classification="Development cross-book context mechanism screen",
            seed=args.seed, paths=args.context_paths, threshold=threshold,
            limits=["Single trained seed", "Fixed-RU training, not joint ES optimization",
                    "No matched-compute or state-of-the-art claim"], stages=[])
        previous_context = ()
        for index, (stage, stage_config) in enumerate(adaptation_configs(banks[0].config)):
            seed = 930000 + 1000*args.seed + 10*index
            print(json.dumps(dict(event="context_stage_start", stage=stage, seed=seed,
                query_paths=args.context_paths, independent_context_books=2)), flush=True)
            query = generate_market_bank(stage_config, args.context_paths, seed)
            row = dict(stage=stage, query_seed=seed, metrics={})
            for label, controller in (("zero_context", AmagoController(experiment.policy)),
                ("time_index_only", AmagoController(experiment.policy,
                    initial_time=2*stage_config.n_steps)),
                ("previous_market_context", AmagoController(experiment.policy,
                    context_banks=previous_context, risk_threshold=threshold)),
                ("conditioned_dh_frozen", policy_controller(direct))):
                values, raw = evaluate_controller(controller, query,
                    device=str(experiment.DEVICE), batch_size=128, zeta=threshold,
                    label=f"{stage}/{label}", progress=True)
                row["metrics"][label] = values
                torch.save(raw, args.output / f"{stage}-{label}.pt")
            # Current-stage training context is created only after before-adaptation evaluation.
            current_context = tuple(generate_market_bank(stage_config, args.context_paths, seed+i)
                for i in (1, 2))
            controller = AmagoController(experiment.policy, context_banks=current_context,
                risk_threshold=threshold)
            values, raw = evaluate_controller(controller, query,
                device=str(experiment.DEVICE), batch_size=128, zeta=threshold,
                label=f"{stage}/current_market_context", progress=True)
            row["metrics"]["current_market_context"] = values
            row["context_decisions"] = controller.context_decisions
            row["context_seeds"] = [seed+1, seed+2]
            torch.save(raw, args.output / f"{stage}-current_market_context.pt")
            context_report["stages"].append(row)
            (args.output / "context-report.json").write_text(json.dumps(context_report, indent=2)+"\n")
            print(json.dumps(dict(event="context_stage_complete", stage=stage,
                es95={name: value["es95"] for name, value in row["metrics"].items()})), flush=True)
            previous_context = current_context


if __name__ == "__main__":
    main()
