"""Reconstruct pilot losses independently and summarize all declared seeds.

This reads completed runs only. Saved DH/band results provide context, not an
equal-extra-training-budget comparison. No checkpoints or input tapes change.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hedging_gym.finance import numpy_ledger
from experiments.qualify_policies import load_bank


def numpy_es(losses, alpha=.95):
    ordered = np.sort(np.asarray(losses, dtype=np.float64))[::-1]
    mass = (1-alpha) * len(ordered)
    whole = int(mass)
    return float((ordered[:whole].sum() + (mass-whole)*ordered[whole])/mass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = dict(scope="Development evidence; no final-test or state-of-the-art claim",
                   rows=[], references=[], maximum_cash_difference=0.)
    for run in args.runs:
        report = json.loads((run / "comparison.json").read_text())
        source = Path(report["arguments"]["source_dir"])
        preset = report["arguments"]["preset"]
        bank = load_bank(source / "development_bank.pt", preset)
        for seed, item in report["seeds"].items():
            for algorithm, result in item["results"].items():
                row = dict(preset=preset, seed=int(seed), algorithm=algorithm,
                           run=str(run.resolve()), evaluation={})
                for deployment, recorded in result["evaluation"].items():
                    tape = torch.load(run / f"seed-{seed}" / f"{algorithm}-{deployment}-tape.pt",
                                      map_location="cpu", weights_only=False)
                    reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
                        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
                    saved_losses = tape["terminal_loss"].double().numpy()
                    difference = float(np.abs(reference["terminal_loss"] - saved_losses).max())
                    summary["maximum_cash_difference"] = max(summary["maximum_cash_difference"], difference)
                    es = numpy_es(saved_losses)
                    np.testing.assert_allclose(es, recorded["es95"], rtol=1e-12, atol=1e-12)
                    row["evaluation"][deployment] = dict(es95=es,
                        ru_at_training_zeta=recorded["ru_at_training_zeta"],
                        tickets_mean=recorded["tickets_mean"], maximum_cash_difference=difference,
                        constraint_violations=recorded["constraint_violations"])
                training = result["training"]
                row["training"] = ({key: training[key] for key in
                    ("total_seconds", "decision_ledger_steps", "terminal_liquidations")}
                    if training else None)
                summary["rows"].append(row)
            # Only the immutable stage reports matching the source stage.
            policy_root = source / "policies"
            stage = 3000
            if preset != "basic":
                policy_root = policy_root / preset
                stage = 1000
            for method in ("dh", "ntb"):
                path = policy_root / f"{method}-seed{seed}" / f"development-{stage}.json"
                saved = json.loads(path.read_text())
                summary["references"].append(dict(preset=preset, seed=int(seed), method=method,
                    es95=saved["evaluation"]["es95"], source=str(path),
                    scope="Previously trained deterministic stage snapshot, without extra updates"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
