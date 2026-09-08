"""Development-only attribution of source work, batch size and ES estimator.

Start every arm at the same original checkpoint. No market, pricing, accounting,
target-update rule or test-set choice is changed. This is a controlled training
recipe comparison, not a new method or native-paper reproduction.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.compare_adapt_aware import load_policy
from experiments.compare_fast_adaptation import rescore_contexts, write_json
from experiments.qualify_adaptation import load_bank
from experiments.summarize_fast_adaptation import es95
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import bank_subset, bank_to, numpy_ledger
from methods.controllers import policy_controller
from methods.meta_pretraining import train_adapt_aware
from methods.training import _report


SEEDS = (7, 17, 29)
TARGETS = ("B", "C", "D")
LOSSES = ("ru", "empirical_es")
BATCHES = (256, 1024)
PATH_WORK = 1_996_800
COMMON_STEPS = 1950


@torch.no_grad()
def source_probe(policy, metadata, banks):
    """Measure threshold tracking on the entire reused source bank, without updates."""
    started = time.perf_counter()
    scorer = deepcopy(policy)
    reports = []
    for task, bank in enumerate(banks):
        scorer.embedding.copy_(scorer.source_embeddings[task])
        metrics, tape = evaluate_controller(policy_controller(scorer), bank,
            device=next(scorer.parameters()).device, batch_size=1024)
        losses = tape["terminal_loss"].double()
        zeta = metadata["source_zetas"][task]
        quantile = float(torch.quantile(losses, bank.config.risk.alpha))
        ru = float(bank.config.risk.loss(losses, losses.new_tensor(zeta)).mean())
        reports.append(dict(task=task, es95=metrics["es95"], stored_zeta=zeta,
            empirical_quantile=quantile, threshold_error=zeta-quantile,
            fraction_above_threshold=float((losses > zeta).double().mean()),
            ru_objective=ru, ru_minus_empirical_es=ru-metrics["es95"]))
    return dict(markets=reports, forward_paths=sum(len(b.spot) for b in banks),
                seconds=time.perf_counter()-started)


def evaluate(args, policy, metadata, output, source_banks):
    report_path = output/"evaluation.json"
    if report_path.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    report = dict(seed=args.seed, loss=args.loss, batch=args.batch,
        source_probe=source_probe(policy, metadata, source_banks), targets={})
    for target in TARGETS:
        calibration = load_bank(args.development/"banks"/f"{target}-cal.pt")
        bank = load_bank(args.development/"banks"/f"{target}-eval.pt")
        scores, work = rescore_contexts(policy, calibration,
            range(len(policy.source_embeddings)), args.device)
        index = min(scores, key=lambda row: row[1])[0]
        scorer = deepcopy(policy)
        with torch.no_grad():
            scorer.embedding.copy_(scorer.source_embeddings[index])
        metrics, tape = evaluate_controller(policy_controller(scorer), bank,
            device=args.device, batch_size=1024)
        torch.save(tape, output/f"{target}-tape.pt")
        report["targets"][target] = dict(metrics=metrics,
            selection=dict(index=index, scores=scores, **work),
            calibration_paths=len(calibration.spot), evaluation_paths=len(bank.spot))
        _report("attribution_evaluation", seed=args.seed, loss=args.loss,
            batch=args.batch, checkpoint=output.name, target=target, es95=metrics["es95"])
    write_json(report_path, report)


def run(args):
    root = args.output/args.loss/f"batch-{args.batch}"/f"seed-{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    original_path = args.source/"adh"/f"seed-{args.seed}"/"pretrained.pt"
    policy, metadata = load_policy(original_path, args.device)
    banks = [bank_to(load_bank(args.source/"banks"/f"source-{i}.pt"), args.device)
             for i in range(len(metadata["source_configs"]))]
    # One arm owns the common initializer evaluation to avoid concurrent writes.
    if args.loss == "empirical_es" and args.batch == 1024:
        evaluate(args, policy, metadata, args.output/"original"/f"seed-{args.seed}", banks)
    milestones = (2,) if args.smoke else sorted({COMMON_STEPS, PATH_WORK//args.batch})
    latest = root/"latest.pt"
    for steps in milestones:
        output = root/f"updates-{steps}"
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output/"policy.pt"
        if checkpoint.exists():
            trained, result = load_policy(checkpoint, args.device)
        else:
            trained, result = train_adapt_aware(deepcopy(policy), metadata, banks,
                seed=args.seed, episodes=steps, inner_updates=0, query_size=args.batch,
                outer_loss=args.loss, outer_lr=1e-4, zeta_lr=3e-4,
                checkpoint_path=latest, resume_from=latest if latest.exists() else None)
            torch.save(dict(policy=trained.state_dict(), metadata=result), checkpoint)
            write_json(output/"training.json", result)
        evaluate(args, trained, result, output, banks)
    _report("attribution_job_complete", seed=args.seed, loss=args.loss, batch=args.batch)


def summarize(args):
    """Independent ES and sampled cash checks; descriptive, not inferential claims."""
    all_rows, audit = [], dict(tapes=0, ledger_paths=0, max_es_error=0., max_cash_error=0.)
    paths = [args.output/"original"/f"seed-{seed}"/"evaluation.json" for seed in SEEDS]
    paths += [args.output/loss/f"batch-{batch}"/f"seed-{seed}"/f"updates-{steps}"/"evaluation.json"
              for loss in LOSSES for batch in BATCHES for seed in SEEDS
              for steps in sorted({COMMON_STEPS,PATH_WORK//batch})]
    for path in paths:
        record = json.loads(path.read_text())
        initial = path.parent.parent.name == "original"
        steps = 0 if initial else int(path.parent.name.removeprefix("updates-"))
        loss, batch, seed = record["loss"], record["batch"], record["seed"]
        for target, row in record["targets"].items():
            tape = torch.load(path.parent/f"{target}-tape.pt", weights_only=False)
            measured = es95(tape["terminal_loss"].double().numpy())
            error = abs(measured-row["metrics"]["es95"])
            if not np.isfinite(measured) or error > 1e-12 or row["metrics"]["constraint_violations"]:
                raise ValueError(f"loss or feasibility audit failed: {path}, {target}")
            bank = load_bank(args.development/"banks"/f"{target}-eval.pt")
            indices = torch.linspace(0,len(bank.spot)-1,32).long()
            sample = bank_subset(bank,indices)
            reconstructed = numpy_ledger(sample.marks.numpy(),tape["positions"][indices].numpy(),
                sample.liability[:,0].numpy(),sample.liability[:,-1].numpy(),sample.config)
            cash_error = float(np.max(np.abs(reconstructed["terminal_loss"]-
                                             tape["terminal_loss"][indices].numpy())))
            if not np.isfinite(cash_error) or cash_error > 2e-6:
                raise ValueError(f"independent cash audit failed: {path}, {target}")
            audit["tapes"] += 1
            audit["ledger_paths"] += len(indices)
            audit["max_es_error"] = max(audit["max_es_error"],error)
            audit["max_cash_error"] = max(audit["max_cash_error"],cash_error)
            all_rows.append(dict(loss="original" if initial else loss,
                batch=0 if initial else batch, steps=steps, seed=seed, target=target,
                es95=measured, gradient_paths=steps*batch))
    groups = []
    keys = sorted({(r["loss"],r["batch"],r["steps"]) for r in all_rows})
    for loss,batch,steps in keys:
        rows = [r for r in all_rows if (r["loss"],r["batch"],r["steps"])==(loss,batch,steps)]
        if {(r["seed"],r["target"]) for r in rows} != {(s,t) for s in SEEDS for t in TARGETS}:
            raise ValueError(f"incomplete attribution group: {loss}, {batch}, {steps}")
        groups.append(dict(loss=loss,batch=batch,updates=steps,gradient_paths=steps*batch,
            mean_es95=float(np.mean([r["es95"] for r in rows])),
            by_seed={s:float(np.mean([r["es95"] for r in rows if r["seed"]==s])) for s in SEEDS},
            by_market={t:float(np.mean([r["es95"] for r in rows if r["target"]==t])) for t in TARGETS}))
    write_json(args.output/"summary.json",dict(groups=groups,rows=all_rows,audit=audit,
        scope="Development only; within-market ES then equal market/seed mean, no selection or confidence claims"))
    _report("attribution_summary",groups=groups,audit=audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("run","summarize"))
    parser.add_argument("--source",type=Path)
    parser.add_argument("--development",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--loss",choices=LOSSES,default="empirical_es")
    parser.add_argument("--batch",type=int,choices=BATCHES,default=1024)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    _report("attribution_start",options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if args.stage == "run": run(args)
    else: summarize(args)


if __name__ == "__main__":
    main()
