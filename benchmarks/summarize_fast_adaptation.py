"""Reconstruct the fast-adaptation screen from saved common-path loss tapes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hedging_gym.evaluation import empirical_es

VARIANTS = dict(adh="adh", geps="geps", belief="belief", srsa_top1="adh",
                srsa_top5="adh", nearest="adh", exhaustive="adh")
SEEDS = (7,17,29)


def es95(values):
    mass = .05 * len(values)
    whole = int(mass)
    ordered_tail = np.partition(-values, whole)
    return -(ordered_tail[:whole].sum() + (mass-whole)*ordered_tail[whole])/mass


def interval(left, right, rng, repeats):
    """Pair market paths and resample the three policy-seed pairs as blocks.

    This is exploratory: three trained seeds are too few to characterize a
    broad training population. The same path indices apply to every seed.
    """
    differences = []
    for _ in range(repeats):
        indices = rng.integers(left.shape[1],size=left.shape[1])
        selected = rng.integers(len(left),size=len(left))
        differences.append(np.mean([es95(left[s,indices])-es95(right[s,indices])
                                    for s in selected]))
    return np.quantile(differences,[.025,.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--bootstrap",type=int,default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(960001)
    report = dict(classification="Three-seed financial-transfer screen, not publication ranking",
        metric="Pooled cost-inclusive terminal ES95; lower is better",
        bootstrap="Paired paths and policy-seed blocks; exploratory with only three seeds",
        bootstrap_seed=960001, bootstrap_repeats=args.bootstrap, targets={})
    for target in ("A","B","C","D"):
        tapes, curves = {}, {}
        for name,family in VARIANTS.items():
            directories = [args.output/family/f"seed-{s}"/target/name for s in SEEDS]
            curves[name] = [json.loads((d/"curve.json").read_text()) for d in directories]
            for budget in (0,10,50,200):
                rows = []
                for directory,curve in zip(directories,curves[name]):
                    raw = torch.load(directory/f"{budget}-tape.pt",map_location="cpu",weights_only=False)
                    loss = raw["terminal_loss"].double().numpy()
                    recorded = next(r for r in curve["milestones"] if r["updates"]==budget)
                    if (not np.isfinite(loss).all() or raw["constraint_violations"].sum()
                            or abs(es95(loss)-recorded["metrics"]["es95"])>1e-12
                            or abs(es95(loss)-empirical_es(torch.from_numpy(loss),.95))>1e-12):
                        raise ValueError(f"tape does not reconcile: {directory}/{budget}")
                    rows.append(loss)
                tapes[name,budget] = np.stack(rows)
        result = {}
        for name in VARIANTS:
            result[name] = {}
            for budget in (0,10,50,200):
                values = [es95(x) for x in tapes[name,budget]]
                baseline = [es95(x) for x in tapes["adh",budget]]
                result[name][budget] = dict(es95_by_seed=values, es95_mean=float(np.mean(values)),
                    improvement_percent=100*(1-np.mean(values)/np.mean(baseline)),
                    improvement_percent_by_seed=(100*(1-np.array(values)/baseline)).tolist())
                if name != "adh" and budget in (0,200):
                    result[name][budget]["difference_ci95"] = interval(
                        tapes[name,budget], tapes["adh",budget],rng,args.bootstrap)
            print(target,name,{b:round(v["improvement_percent"],2)
                               for b,v in result[name].items()},flush=True)
        report["targets"][target] = result
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
