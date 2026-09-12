"""Fresh A-B-A development evidence from existing, separately trained baselines.

This reuses checkpoints without retraining their source policies. Pretraining
budgets differ and are reported; this is not a compute-matched method ranking.
Each stage changes only market parameters, with paired paths across methods.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch

from hedging_gym.benchmark import adaptation_configs
from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import generate_market_bank
from methods.adaptation import AdaptationUpdater, TaskEmbeddedPolicy
from methods.controllers import policy_controller
from methods.policies import DirectDHPolicy
from methods.training import _report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage-updates", type=int, default=200)
    parser.add_argument("--train-paths", type=int, default=4096)
    parser.add_argument("--eval-paths", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    direct_path = args.source / "policies" / f"dh-seed{args.seed}" / "latest.pt"
    direct_saved = torch.load(direct_path, map_location="cpu", weights_only=False)
    config = config_from_dict(direct_saved["config"])
    direct = DirectDHPolicy(config, hidden=direct_saved["options"]["hidden"]).to(args.device)
    direct.load_state_dict(direct_saved["policy"])
    embedded_path = args.source / "adaptation" / f"seed-{args.seed}" / "multitask-policy.pt"
    embedded_saved = torch.load(embedded_path, map_location="cpu", weights_only=False)
    metadata = embedded_saved["metadata"]
    embedded = TaskEmbeddedPolicy(config, n_tasks=len(metadata["source_configs"]),
        embedding_dim=metadata["options"]["embedding_dim"],
        hidden=metadata["options"]["hidden"]).to(args.device)
    embedded.load_state_dict(embedded_saved["policy"])
    embedded.prepare_adaptation()
    policies = dict(dh_frozen=deepcopy(direct), dh_finetune=direct,
                    embedding_frozen=deepcopy(embedded), embedding_adapt=embedded)
    updaters = {
        "dh_finetune": AdaptationUpdater(direct, metadata=dict(zeta=float(direct_saved["zeta"])),
            seed=args.seed+800000, updates=args.stage_updates, batch_size=128),
        "embedding_adapt": AdaptationUpdater(embedded, metadata=metadata,
            seed=args.seed+800000, updates=args.stage_updates, batch_size=128),
    }
    report = dict(classification="Fresh development baseline qualification; no AMAGO finance claim",
        seed=args.seed, source_checkpoints=dict(dh=str(direct_path), embedding=str(embedded_path)),
        source_pretraining_updates=dict(dh=direct_saved["step"], embedding=metadata["options"]["updates"]),
        stage_updates=args.stage_updates, train_paths=args.train_paths, eval_paths=args.eval_paths,
        observation_contract="All existing observed market parameters retained; no regime ID added",
        stages=[])
    report_path = args.output / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if any(previous[key] != report[key] for key in
               ("seed", "stage_updates", "train_paths", "eval_paths", "source_checkpoints")):
            raise ValueError("resume requires the original source and experiment settings")
        report = previous
        if report["stages"]:
            last_stage = report["stages"][-1]["stage"]
            for name, updater in updaters.items():
                saved = torch.load(args.output / f"{name}-{last_stage}.pt",
                    map_location="cpu", weights_only=False)
                updater.load_state_dict(saved["state"])
    _report("adaptation_review_start", **report)
    for stage_index, (stage, current) in enumerate(adaptation_configs(config)):
        if stage_index < len(report["stages"]):
            continue
        stage_seed = 810000 + 1000*args.seed + 10*stage_index
        row = dict(stage=stage, config=asdict(current), seeds=dict(before=stage_seed,
            train=stage_seed+1, after=stage_seed+2), before={}, after={})
        for phase in ("before", "after"):
            if phase == "after":
                training = generate_market_bank(current, args.train_paths, stage_seed+1)
                row["updates"] = {}
                for name, updater in updaters.items():
                    updater.checkpoint_path = args.output / f"{name}-{stage}.pt"
                    row["updates"][name] = updater(training)
            evaluation = generate_market_bank(current, args.eval_paths,
                stage_seed if phase == "before" else stage_seed+2)
            for name, policy in policies.items():
                metrics, raw = evaluate_controller(policy_controller(policy), evaluation,
                    device=args.device, batch_size=1024, label=f"{name}/{stage}/{phase}", progress=True)
                row[phase][name] = metrics
                torch.save(raw, args.output / f"{name}-{stage}-{phase}.pt")
        report["stages"].append(row)
        report_path.write_text(json.dumps(report, indent=2)+"\n")
        _report("adaptation_review_stage_complete", seed=args.seed, market_stage=stage,
            before={name: value["es95"] for name, value in row["before"].items()},
            after={name: value["es95"] for name, value in row["after"].items()})


if __name__ == "__main__":
    main()
