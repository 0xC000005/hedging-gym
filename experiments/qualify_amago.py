"""Functional qualification of official AMAGO on its MetaFrozenLake task.

Uses the source agent, transformer, memory environment and replay pipeline.
The shortened schedule is an execution check, not an ICLR table reproduction.
Install AMAGO in its own environment; its Gymnasium dependency differs from
the common hedging environment. All output is written outside the repository.
"""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

import amago
from amago import cli_utils
from amago.envs import AMAGOEnv
from amago.envs.builtin.toy_gym import MetaFrozenLake
from amago.loading import DiskTrajDataset
from amago.nets.transformer import VanillaAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    # Public example's default task geometry and ten consecutive attempts.
    horizon = 5 * 5 * 10
    config = {}
    trajectory = cli_utils.switch_traj_encoder(config, arch="transformer",
        memory_size=128, layers=3, attention_type=VanillaAttention)
    timestep = cli_utils.switch_tstep_encoder(config, arch="ff", n_layers=1,
        d_hidden=128, d_output=64, normalize_inputs=False)
    exploration = cli_utils.switch_exploration(config, strategy="egreedy",
        eps_start=1.0, eps_end=.05, steps_anneal=1_000_000, randomize_eps=True)
    agent = cli_utils.switch_agent(config, "agent", tau=.004)
    cli_utils.use_config(config)
    make_env = lambda: AMAGOEnv(MetaFrozenLake(size=5, k_episodes=10),
        env_name="meta_frozen_lake_k10_5x5_easy_reset")
    dataset = DiskTrajDataset(dset_root=str(args.output), dset_name="native",
        dset_max_size=12_500)
    experiment = amago.Experiment(run_name="native", ckpt_base_dir=str(args.output),
        max_seq_len=horizon, traj_save_len=horizon, dataset=dataset,
        make_train_env=make_env, make_val_env=make_env,
        agent_type=agent, exploration_wrapper_type=exploration,
        tstep_encoder_type=timestep, traj_encoder_type=trajectory,
        dloader_workers=0, log_to_wandb=False, epochs=args.epochs,
        parallel_actors=4, batch_size=4, train_timesteps_per_epoch=horizon,
        train_batches_per_epoch=args.batches, val_interval=1,
        val_timesteps_per_epoch=horizon * 2, ckpt_interval=1, env_mode="sync")
    print(json.dumps(dict(event="qualification_start", seed=args.seed,
        expected_updates=args.epochs * args.batches,
        expected_training_transitions=args.epochs * horizon * 4,
        source="UT-Austin-RPL/amago@54c25ab6da9371614c47352f569a56c0fe938d3b",
        classification="Native functional qualification, not paper reproduction")), flush=True)
    experiment.start()
    before = experiment.evaluate_test(make_env, timesteps=1000)
    experiment.learn()
    after = experiment.evaluate_test(make_env, timesteps=1000)
    record = dict(classification="Native functional qualification, not paper reproduction",
        seed=args.seed, epochs=args.epochs, batches_per_epoch=args.batches,
        source_commit="54c25ab6da9371614c47352f569a56c0fe938d3b",
        task="Official MetaFrozenLake 5x5, ten attempts, default task dynamics",
        source_changes=["shortened schedule", "4 instead of 32 actors", "batch size 4",
                        "official VanillaAttention instead of optional FlashAttention"],
        elapsed_seconds=time.perf_counter()-started, before=before, after=after)
    (args.output / "qualification.json").write_text(json.dumps(record, indent=2, default=str)+"\n")
    print(json.dumps(dict(event="qualification_complete", **record), default=str), flush=True)


if __name__ == "__main__":
    main()
