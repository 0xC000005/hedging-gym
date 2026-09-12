"""Learning-rate check and first-order adaptation-aware pretraining comparison.

Development B/C/D choose settings. Frozen E/F/G/H combinations are evaluated
only after selection. Source, development and final data have separate roles;
all methods share the existing Heston pricing, actions and accounting.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.compare_fast_adaptation import market_config, rescore_contexts, write_json
from experiments.compare_update_capacity import matched_updater, path_usage
from experiments.qualify_adaptation import load_bank
from hedging_gym.config import config_from_dict
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import BANK_FIELDS, bank_subset, bank_to, generate_market_bank
from methods.adaptation import TaskEmbeddedPolicy
from methods.checkpoints import save_checkpoint
from methods.controllers import policy_controller
from methods.training import _report, _sync, rollout


TEST_MARKETS = dict(E=(.045,.075,3.5,.35,-.55), F=(.075,.05,2.5,.32,-.65),
                    G=(.035,.025,2.7,.28,-.4), H=(.065,.085,3.2,.38,-.6))
SEEDS = (7, 17, 29)
FAMILIES = ("original", "ordinary", "meta")
MODES = ("embedding", "finetune")
RATES = (1e-3, 3e-4, 1e-4)
BUDGETS = (0, 10, 50, 200)
PATH_COUNTS = (256, 1024, 4096)


def load_policy(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    metadata, weights = saved["metadata"], saved["policy"]
    policy = TaskEmbeddedPolicy(config_from_dict(metadata["config"]),
        n_tasks=len(metadata["source_configs"]),
        embedding_dim=metadata["options"]["embedding_dim"],
        hidden=metadata["options"]["hidden"]).to(device)
    policy.load_state_dict(weights)
    policy.prepare_adaptation()
    return policy, metadata


def checkpoint_path(args):
    if args.family == "original":
        return args.source / "adh" / f"seed-{args.seed}" / "pretrained.pt"
    return args.output / "pretraining" / args.family / f"seed-{args.seed}" / "pretrained.pt"


def train(args):
    from methods.meta_pretraining import train_adapt_aware
    if args.family == "original":
        raise ValueError("original checkpoints are reused, not retrained")
    output = args.output / "pretraining" / args.family / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pretrained.pt").exists():
        _report("pretraining_already_complete", family=args.family, seed=args.seed)
        return
    policy, metadata = load_policy(args.source / "adh" / f"seed-{args.seed}" / "pretrained.pt", args.device)
    banks = [bank_to(load_bank(args.source / "banks" / f"source-{i}.pt"), args.device)
             for i in range(len(metadata["source_configs"]))]
    episodes = 1950 if args.family == "ordinary" else 600
    policy, metadata = train_adapt_aware(policy, metadata, banks, seed=args.seed,
        episodes=2 if args.smoke else episodes,
        inner_updates=0 if args.family == "ordinary" else 5,
        checkpoint_path=output / "latest.pt", resume_from=(output / "latest.pt")
        if (output / "latest.pt").exists() else None)
    torch.save(dict(policy=policy.state_dict(), metadata=metadata), output / "pretrained.pt")
    write_json(output / "pretraining.json", metadata)


def prepare_banks(args):
    output = args.output / "banks"
    output.mkdir(parents=True, exist_ok=True)
    for index, (name, values) in enumerate(TEST_MARKETS.items()):
        config = market_config(values)
        for role, paths, base_seed in (("train",4096,1010000),("cal",1024,1020000),("eval",8192,1030000)):
            path = output / f"{name}-{role}.pt"
            seed = base_seed+index
            if path.exists():
                old = torch.load(path, map_location="cpu", weights_only=False)
                if config_from_dict(old["config"]) != config or old["seed"] != seed or len(old["spot"]) != paths:
                    raise ValueError("bank differs from frozen settings")
                continue
            started = time.perf_counter()
            _report("bank_start", market=name, role=role, paths=paths, seed=seed, device=args.device)
            bank = generate_market_bank(config, paths, seed, device=args.device)
            _sync(args.device)
            torch.save(dict(config=asdict(config), seed=seed, generation_seconds=time.perf_counter()-started,
                **{key:getattr(bank,key).cpu() for key in BANK_FIELDS}), path)
            _report("bank_complete", market=name, role=role, seconds=time.perf_counter()-started)


def curve(args, policy, metadata, target, mode, rate, train_bank, calibration, evaluation, directory,
          *, budgets=BUDGETS):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return
    device = torch.device(args.device)
    train_bank = bank_to(train_bank, device)
    checkpoint = directory / "latest.pt"
    scores, work = rescore_contexts(policy, calibration, range(len(policy.source_embeddings)), device)
    index = min(scores, key=lambda row: row[1])[0]
    initial = policy.source_embeddings[index].detach().clone()
    count = min(1024, len(train_bank.spot))
    scorer = deepcopy(policy)
    with torch.no_grad():
        scorer.embedding.copy_(initial)
        started = time.perf_counter()
        initial_losses = rollout(scorer, bank_subset(train_bank, slice(0,count)))["terminal_loss"]
        zeta = torch.quantile(initial_losses, train_bank.config.risk.alpha)
        _sync(device)
        initialization_seconds = time.perf_counter()-started
    seed = args.seed+2000
    updater = matched_updater(policy, metadata, train_bank, initial, zeta,
                              mode=mode, seed=seed, learning_rate=rate,
                              updates=5 if budgets == (0, 5) else 10)
    report = dict(family=args.family, policy_seed=args.seed, target=target, mode=mode,
        rate=rate, train_paths=len(train_bank.spot), config=asdict(train_bank.config),
        minibatch_seed=seed+100003, calibration_paths=len(calibration.spot),
        selection=dict(index=index, scores=scores, **work),
        threshold_paths=count, threshold_seconds=initialization_seconds,
        checkpoint_source=str(checkpoint_path(args)), milestones=[])
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        updater.load_state_dict(saved["updater"])
        report = saved["report"]
    recorded = {record["updates"] for record in report["milestones"]}
    for budget in budgets:
        if budget in recorded:
            continue
        if updater.completed_steps > budget:
            raise ValueError("saved checkpoint has passed an unsaved milestone")
        while updater.completed_steps < budget:
            updater(train_bank)
        metrics, tape = evaluate_controller(policy_controller(updater.policy), evaluation,
            device=device, batch_size=1024, label=f"{args.family}/{target}/{mode}/{rate}/{budget}")
        record = dict(updates=budget, metrics=metrics,
            unique_target_paths=len(calibration.spot)+path_usage(seed,len(train_bank.spot),256,budget,count),
            gradient_episodes=budget*256,
            adaptation_seconds=sum(r["elapsed_seconds"] for r in updater.history))
        report["milestones"].append(record)
        torch.save(tape, directory / f"{budget}-tape.pt")
        payload = dict(step=budget, updater=updater.state_dict(), report=report)
        save_checkpoint(directory / f"{budget}-policy.pt", payload)
        save_checkpoint(checkpoint, payload)
        write_json(directory / "curve.json", report)
        _report("adapt_point", family=args.family, seed=args.seed, target=target, mode=mode,
            rate=rate, train_paths=len(train_bank.spot), updates=budget, es95=metrics["es95"])
    write_json(directory / "complete.json", report)


def compare(args):
    testing = args.stage == "test"
    selection = json.loads((args.output / "selection.json").read_text()) if testing else None
    policy, metadata = load_policy(checkpoint_path(args), args.device)
    targets = TEST_MARKETS if testing else ("B","C","D")
    directory = args.output / args.stage / args.family / f"seed-{args.seed}"
    bank_dir = args.output / "banks" if testing else args.development / "banks"
    for target in targets:
        train_bank, cal, evaluation = (load_bank(bank_dir / f"{target}-{role}.pt")
                                       for role in ("train","cal","eval"))
        for mode in MODES:
            rates = [selection[args.family][mode]["rate"]] if testing else RATES
            for rate in rates:
                for count in PATH_COUNTS if testing else (4096,):
                    curve(args,policy,metadata,target,mode,rate,bank_subset(train_bank,slice(0,count)),
                          cal,evaluation,directory/target/mode/f"lr-{rate:g}"/f"paths-{count}")
    _report("comparison_complete", stage_name=args.stage, family=args.family, seed=args.seed)


def select(args):
    """One rate per family/mode, never one selected per test market or seed."""
    selection = {}
    for family in FAMILIES:
        selection[family] = {}
        for mode in MODES:
            scores = {}
            for rate in RATES:
                ratios = []
                for seed in SEEDS:
                    for target in ("B","C","D"):
                        curve_dir = args.output/"develop"/family/f"seed-{seed}"/target/mode/f"lr-{rate:g}"/"paths-4096"
                        data = json.loads((curve_dir/"complete.json").read_text())
                        anchor = json.loads((args.development/f"seed-{seed}"/target/"embedding"/"curve.json").read_text())
                        denominator = anchor["milestones"][0]["metrics"]["es95"]
                        ratios.append(next(r["metrics"]["es95"] for r in data["milestones"] if r["updates"]==10)/denominator)
                scores[rate] = float(np.mean(ratios))
            best = min(scores, key=scores.get)
            selection[family][mode] = dict(rate=best, normalized_development_es_at_10=scores[best],
                                          candidates=scores)
    path = args.output/"selection.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(selection)):
        raise ValueError("a frozen selection already differs")
    write_json(path, selection)
    _report("selection_frozen", selection=selection)


def diagnose(args):
    """Development-only check of the five-step recipe used during meta-training.

    This is a fixed diagnostic, not another tuning candidate. Context scoring,
    threshold initialization and minibatch streams remain unchanged.
    """
    if args.family == "original":
        raise ValueError("the recipe diagnostic compares meta and ordinary continuations")
    policy, metadata = load_policy(checkpoint_path(args), args.device)
    for target in ("B", "C", "D"):
        banks = [load_bank(args.development/"banks"/f"{target}-{role}.pt")
                 for role in ("train", "cal", "eval")]
        directory = args.output/"diagnosis"/args.family/f"seed-{args.seed}"/target
        curve(args, policy, metadata, target, "finetune", 3e-4, *banks, directory, budgets=(0,5))
    _report("diagnosis_complete", family=args.family, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("banks","train","develop","select","test","diagnose"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--family", choices=FAMILIES, default="original")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    _report("adapt_aware_start", options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if args.stage == "banks": prepare_banks(args)
    elif args.stage == "train": train(args)
    elif args.stage == "select": select(args)
    elif args.stage == "diagnose": diagnose(args)
    else: compare(args)


if __name__ == "__main__":
    main()
