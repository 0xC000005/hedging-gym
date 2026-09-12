"""Train/reload DH, learned bands and hybrid policies on common saved banks.

The input banks are trusted local artifacts. This records development outcomes;
final comparisons need a separately generated bank after settings are frozen.
"""

import argparse
import json
from pathlib import Path

import torch

from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines._shared.training import _report
from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.baselines.deep_hedging import train as train_dh
from hedging_gym.baselines.hpo import HybridPolicy, train_hybrid
from hedging_gym.baselines.no_transaction_band import NoTransactionBandPolicy
from hedging_gym.baselines.no_transaction_band import train as train_ntb
from hedging_gym.environment.benchmark import operational_config
from hedging_gym.environment.config import config_from_dict
from hedging_gym.environment.finance import BANK_FIELDS, MarketBank
from hedging_gym.evaluation import evaluate_controller

POLICIES = {"dh": DirectDHPolicy, "ntb": NoTransactionBandPolicy}


def load_bank(path, preset):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_dict(saved["config"])
    if preset != "basic":
        # Execution fees do not change the simulated state or clean prices.
        config = operational_config(config, preset)
    return MarketBank(*(saved[key] for key in BANK_FIELDS), config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--methods", nargs="+", choices=("dh", "ntb", "hpo"), default=["dh", "ntb", "hpo"])
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", choices=("basic", "operational_fixed", "operational_minimum_fee"), default="basic")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train = load_bank(args.run_dir / "train_bank.pt", args.preset)
    development = load_bank(args.run_dir / "development_bank.pt", args.preset)
    output = args.run_dir / "policies"
    if args.preset != "basic":
        output = output / args.preset
    for seed in args.seeds:
        for method in args.methods:
            directory = output / f"{method}-seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory / "latest.pt"
            options = dict(seed=seed, updates=args.updates, batch_size=args.batch_size,
                hidden=(64, 64), device=args.device, checkpoint_path=checkpoint,
                checkpoint_every=200, resume_from=checkpoint if checkpoint.exists() else None)
            policy, metadata = (train_hybrid(train, **options) if method == "hpo"
                                else {'dh': train_dh, 'ntb': train_ntb}[method](train, **options))
            # Qualification evaluates the disk artifact, not a live training object.
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            reloaded = (HybridPolicy if method == "hpo" else POLICIES[method])(
                train.config, hidden=(64, 64)).to(device=args.device, dtype=train.spot.dtype)
            reloaded.load_state_dict(saved["policy"])
            reloaded.eval()
            metrics, tape = evaluate_controller(policy_controller(reloaded), development,
                device=args.device, batch_size=1024, label=f"{method}/development",
                zeta=metadata["zeta"], progress=True)
            torch.save(saved, directory / f"step-{args.updates}.pt")
            torch.save(tape, directory / f"development-{args.updates}-tape.pt")
            report = dict(scope="Development qualification; not a final ranking",
                preset=args.preset, arguments={key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}, training=metadata, evaluation=metrics)
            (directory / f"development-{args.updates}.json").write_text(json.dumps(report, indent=2)+"\n")
            _report("policy_qualification", method=method, seed=seed, preset=args.preset,
                checkpoint=str(checkpoint), es95=metrics["es95"], violations=metrics["constraint_violations"])


if __name__ == "__main__":
    main()
