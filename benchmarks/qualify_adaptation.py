"""Train task embeddings and compare chronological adaptation on shared banks.

This is a development qualification, not a paper reproduction or final ranking.
All output, including resumable checkpoints, belongs outside the repository.
"""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch

from hedging_gym.baselines._shared.adaptation import AdaptationUpdater
from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines._shared.training import _report
from hedging_gym.baselines.adaptive_deep_hedging import train_multitask
from hedging_gym.baselines.deep_hedging import DirectDHPolicy
from hedging_gym.environment.benchmark import adaptation_configs
from hedging_gym.environment.config import config_from_dict
from hedging_gym.environment.finance import (
    BANK_FIELDS,
    MarketBank,
    generate_market_bank,
)
from hedging_gym.evaluation import evaluate_controller


def load_bank(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_dict(saved["config"])
    return MarketBank(*(saved[key] for key in BANK_FIELDS), config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--stage-updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stage-train-paths", type=int, default=8192)
    parser.add_argument("--stage-eval-paths", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--dh-checkpoint", type=Path)
    parser.add_argument("--pretrain-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.run_dir / "adaptation" / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    source = load_bank(args.run_dir / "train_bank.pt")
    config = source.config
    second_config = replace(config, market=replace(config.market, v0=.032, theta=.032))
    second_path = args.run_dir / "adaptation" / "source-nearby.pt"
    if second_path.exists():
        second = load_bank(second_path)
    else:
        _report("source_bank_start", paths=len(source.spot), seed=101101,
                market=asdict(second_config.market))
        second = generate_market_bank(second_config, len(source.spot), 101101)
        torch.save(dict(config=asdict(second.config), seed=101101,
                        **{key: getattr(second, key) for key in BANK_FIELDS}), second_path)
        _report("source_bank_complete", paths=len(source.spot))
    checkpoint = output / "multitask-latest.pt"
    policy, metadata = train_multitask((source, second), seed=args.seed,
        updates=args.updates, batch_size=args.batch_size, hidden=(64, 64), device=args.device,
        checkpoint_path=checkpoint, checkpoint_every=100,
        resume_from=checkpoint if checkpoint.exists() else None)
    torch.save(dict(policy=policy.state_dict(), metadata=metadata), output / "multitask-policy.pt")
    (output / "pretraining.json").write_text(json.dumps(metadata, indent=2)+"\n")
    development = load_bank(args.run_dir / "development_bank.pt")
    pretraining_metrics, raw = evaluate_controller(policy_controller(policy), development,
        device=args.device, batch_size=1024, label="multitask/development", progress=True)
    torch.save(raw, output / "development-tape.pt")
    (output / "development.json").write_text(json.dumps(pretraining_metrics, indent=2)+"\n")
    _report("pretraining_development", seed=args.seed, metrics=pretraining_metrics)
    if args.pretrain_only:
        return

    policies = dict(embedding_frozen=deepcopy(policy), embedding_adapt=policy)
    updaters = dict(embedding_adapt=AdaptationUpdater(policy, metadata=metadata,
        seed=args.seed+1000, updates=args.stage_updates, batch_size=args.batch_size,
        checkpoint_path=output / "embedding-adaptation-latest.pt"))
    if args.dh_checkpoint:
        saved = torch.load(args.dh_checkpoint, map_location="cpu", weights_only=False)
        direct = DirectDHPolicy(config, hidden=saved["options"]["hidden"]).to(args.device)
        direct.load_state_dict(saved["policy"])
        policies.update(dh_frozen=deepcopy(direct), dh_finetune=direct)
        updaters["dh_finetune"] = AdaptationUpdater(direct,
            metadata=dict(zeta=float(saved["zeta"])), seed=args.seed+1000,
            updates=args.stage_updates, batch_size=args.batch_size,
            checkpoint_path=output / "finetune-adaptation-latest.pt")
    result = dict(classification="Development adaptation qualification, not final comparison",
                  policy_seed=args.seed, pretraining=metadata, stages=[])
    stage_seed = 31001
    for index, (stage, current) in enumerate(adaptation_configs(config)):
        # No future stage is generated or shown to an updater early.
        _report("adaptation_stage", name=stage, market=asdict(current.market))
        before = generate_market_bank(current, args.stage_eval_paths, stage_seed+10*index)
        before_metrics = {}
        for name, controller in policies.items():
            metrics, tape = evaluate_controller(policy_controller(controller), before,
                device=args.device, batch_size=1024, label=name+"/"+stage+"/before", progress=True)
            before_metrics[name] = metrics
            torch.save(tape, output / f"{stage}-{name}-before.pt")
        train = generate_market_bank(current, args.stage_train_paths, stage_seed+10*index+1)
        records = {}
        for name, update in updaters.items():
            update.checkpoint_path = output / f"{name}-{stage}-latest.pt"
            records[name] = update(train)
        after = generate_market_bank(current, args.stage_eval_paths, stage_seed+10*index+2)
        after_metrics = {}
        for name, controller in policies.items():
            metrics, tape = evaluate_controller(policy_controller(controller), after,
                device=args.device, batch_size=1024, label=name+"/"+stage+"/after", progress=True)
            after_metrics[name] = metrics
            torch.save(tape, output / f"{stage}-{name}-after.pt")
        result["stages"].append(dict(stage=stage, config=asdict(current),
            train_seed=stage_seed+10*index+1, before_seed=stage_seed+10*index,
            after_seed=stage_seed+10*index+2, before=before_metrics, after=after_metrics,
            updates=records))
        (output / "adaptation.json").write_text(json.dumps(result, indent=2)+"\n")
        _report("adaptation_stage_complete", name=stage, before=before_metrics, after=after_metrics)


if __name__ == "__main__":
    main()
