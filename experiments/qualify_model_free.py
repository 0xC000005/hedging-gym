"""Source-reasoned model-free training on saved common training/development banks.

Uses the native replay intensity (32 samples per new transition) and Adam1e-4.
The optional dense labels telescope to the same global terminal ES objective.
This is development qualification, not a published-benchmark reproduction.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import BANK_FIELDS, MarketBank
from methods.controllers import policy_controller
from methods.model_free import train_model_free


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="directory containing train_bank.pt and development_bank.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("hull_rl", "exdrl"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reward-labels", choices=("dense", "terminal"), default="dense")
    parser.add_argument("--resume", type=Path,
                        help="trusted local complete training checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def bank(name):
        saved = torch.load(args.run_dir/name, map_location="cpu", weights_only=False)
        return MarketBank(*(saved[key] for key in BANK_FIELDS), config_from_dict(saved["config"]))

    training = bank("train_bank.pt")
    # Eight collected episodes * dates *32 samples per insertion / batch128.
    gradient_steps = 2*training.config.n_steps
    policy, metadata = train_model_free(args.method, training,
        seed=args.seed, updates=args.updates, batch_size=128, collection_batch_size=8,
        hidden=(64,64), device=args.device, gradient_steps=gradient_steps,
        learning_rate=1e-4, critic_learning_rate=1e-4,
        dense_rewards=args.reward_labels == "dense", checkpoint_every=100,
        checkpoint_path=args.output_dir/"latest.pt", resume_from=args.resume)
    development = bank("development_bank.pt")
    metrics, tape = evaluate_controller(policy_controller(policy), development,
        device=args.device, batch_size=1024, zeta=metadata["zeta"], progress=True,
        label=f"{args.method}: development qualification")
    torch.save(dict(policy=policy.state_dict(), config=asdict(training.config), metadata=metadata),
               args.output_dir/"policy.pt")
    torch.save(tape, args.output_dir/"development.pt")
    report = dict(scope="development qualification, not final comparison", metadata=metadata,
                  metrics=metrics, training_bank=str(args.run_dir/"train_bank.pt"),
                  development_bank=str(args.run_dir/"development_bank.pt"))
    (args.output_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
