"""Stock PPO on shared Heston data with complete-episode ES training batches.

PPO itself is unmodified. Global RU threshold is fixed within a training phase
and fitted to training-only sampled-policy losses between phases. This is an
independent common-objective control, not a Hull or EX-DRL reproduction.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import BANK_FIELDS, MarketBank, bank_subset, numpy_ledger
from methods.sb3 import SB3HedgingVecEnv, sb3_controller


def load_bank(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return MarketBank(*(saved[key] for key in BANK_FIELDS), config_from_dict(saved["config"]))


class Progress(BaseCallback):
    def __init__(self, total, episode_steps):
        super().__init__()
        self.total, self.episode_steps = total, episode_steps
        self.started = time.perf_counter()

    def _on_step(self):
        if self.num_timesteps % self.episode_steps == 0:
            elapsed = time.perf_counter() - self.started
            print(json.dumps(dict(stage="PPO rollout", transitions=self.num_timesteps,
                target=self.total, seconds=elapsed,
                eta_seconds=elapsed*(self.total-self.num_timesteps)/self.num_timesteps)), flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--rollouts-per-phase", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    training = load_bank(args.run_dir / "train_bank.pt")
    development = load_bank(args.run_dir / "development_bank.pt")
    calibration = bank_subset(training, slice(0, min(4096, len(training.spot))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_rollout = args.envs * training.config.n_steps
    total = args.phases * args.rollouts_per_phase * steps_per_rollout
    recipe = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        config=asdict(training.config), transitions=total,
        episodes=total//training.config.n_steps, episodes_per_rollout=args.envs,
        nominal_tail_episodes_per_rollout=args.envs*(1-training.config.risk.alpha),
        gamma=1., gae_lambda=1., learning_rate=3e-4, hidden=[64,64],
        log_std_init=-1., device="cpu", training="uniform batches without replacement within a batch from saved training bank")
    (args.output_dir / "recipe.json").write_text(json.dumps(recipe, indent=2)+"\n")
    print(json.dumps(recipe), flush=True)
    env = SB3HedgingVecEnv(training, args.envs)
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
        n_steps=training.config.n_steps, batch_size=min(1024, steps_per_rollout),
        n_epochs=args.epochs, gamma=1., gae_lambda=1., learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64]), log_std_init=-1.))
    progress = Progress(total, steps_per_rollout)
    records = []
    for phase in range(args.phases+1):
        _, sample = evaluate_controller(sb3_controller(model, deterministic=False), calibration,
                                       batch_size=512, mode_seed=91001+phase)
        zeta = float(torch.quantile(sample["terminal_loss"], training.config.risk.alpha))
        record = dict(phase=phase, transitions=model.num_timesteps, zeta=zeta, scores={})
        for deterministic in (True, False):
            name = "greedy" if deterministic else "sampled"
            metrics, tape = evaluate_controller(sb3_controller(model, deterministic=deterministic),
                development, batch_size=512, mode_seed=30001, zeta=zeta)
            reference = numpy_ledger(development.marks.numpy(), tape["positions"].numpy(),
                development.liability[:,0].numpy(), development.liability[:,-1].numpy(), development.config)
            error = float(np.max(np.abs(reference["terminal_loss"]-tape["terminal_loss"].numpy())))
            np.testing.assert_allclose(reference["terminal_loss"], tape["terminal_loss"], atol=2e-6, rtol=0)
            record["scores"][name] = dict(metrics=metrics, cash_error=error)
            torch.save(tape, args.output_dir / f"phase{phase}-{name}.pt")
        model.save(args.output_dir / f"phase{phase}")
        records.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        if phase == args.phases:
            break
        # Every previous rollout ended at settlement; SB3 has already autoreset
        # to t=0. The global reward threshold changes before the next action.
        env.risk_threshold = zeta
        model.learn(args.rollouts_per_phase*steps_per_rollout,
                    reset_num_timesteps=False, callback=progress)
    reloaded = PPO.load(args.output_dir / f"phase{args.phases}", device="cpu")
    _, loaded_tape = evaluate_controller(sb3_controller(reloaded), development,
                                        batch_size=512, mode_seed=30001)
    original = torch.load(args.output_dir / f"phase{args.phases}-greedy.pt", weights_only=False)
    torch.testing.assert_close(loaded_tape["terminal_loss"], original["terminal_loss"], rtol=0, atol=0)
    print("Final checkpoint reload: identical greedy terminal losses", flush=True)
    env.close()


if __name__ == "__main__":
    main()
