"""Frozen-policy confirmation on one new unchanged-Heston development bank."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from experiments.qualify_counterfactual import sampled_controller, _load_policy
from experiments.qualify_policies import load_bank
from hedging_gym.evaluation import evaluate_controller
from hedging_gym.finance import numpy_ledger
from methods.controllers import policy_controller
from methods.training import POLICIES, _report


def es_rows(losses, alpha=.95):
    """Empirical ES by partial sorting; preserve fractional boundary mass."""
    mass = (1 - alpha) * losses.shape[-1]
    whole = int(mass)
    first = losses.shape[-1] - whole - 1
    ordered = np.partition(losses, first, axis=-1)
    return (ordered[..., first + 1:].sum(-1) + (mass - whole) * ordered[..., first]) / mass


def paired_intervals(losses, seed, repeats=1000):
    """Resample common path indices; intervals condition on trained policies."""
    generator = np.random.default_rng(seed)
    values = {name: [] for name in losses}
    paths = len(losses["all_mode"])
    for offset in range(0, repeats, 32):
        indices = generator.integers(paths, size=(min(32, repeats - offset), paths))
        for name, sample in losses.items():
            values[name].extend(es_rows(sample[indices]).tolist())
    reference = np.asarray(values["all_mode"])
    return {name: dict(delta_es95=float(es_rows(losses["all_mode"]) - es_rows(sample)),
        bootstrap95=np.quantile(reference - np.asarray(values[name]), [.025, .975]).tolist())
        for name, sample in losses.items() if name != "all_mode"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    raw_bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    report = dict(scope="Frozen-policy fresh development confirmation, not final publication test",
        bank=str(args.bank.resolve()), bank_seed=raw_bank["seed"], paths=raw_bank["n_paths"],
        confidence_scope="1000 paired path-bootstrap resamples per fixed trained policy; not training-seed uncertainty",
        comparison="all_mode minus comparator; lower/negative is better", cases={})
    for case, preset in (("basic", "basic"), ("fixed", "operational_fixed")):
        run = args.run_root / f"trajectory-{case}"
        controls = args.run_root / f"joint-{case}"
        specification = json.loads((run / "comparison.json").read_text())
        bank = load_bank(args.bank, preset)
        case_report = {}
        report["cases"][case] = case_report
        for seed_text, item in specification["seeds"].items():
            seed = int(seed_text)
            directory = args.output / case / f"seed-{seed}"
            directory.mkdir(parents=True)
            strong = "dh" if case == "basic" else "ntb"
            checkpoints = dict(source=Path(item["source_checkpoint"]),
                all_mode=run / f"seed-{seed}/all_mode/latest.pt",
                sampled=run / f"seed-{seed}/sampled/latest.pt",
                full_hpo=controls / f"seed-{seed}/full_hpo/latest.pt")
            checkpoints[strong] = controls / f"seed-{seed}/{strong}/latest.pt"
            seed_report = dict(results={})
            case_report[seed_text] = seed_report
            losses = {}
            for name, path in checkpoints.items():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["config"] != asdict(bank.config):
                    raise ValueError("checkpoint and unchanged finance confirmation configuration differ")
                if name == strong:
                    policy = POLICIES[name](bank.config, hidden=saved["options"]["hidden"]).to(
                        device=args.device, dtype=bank.spot.dtype)
                    policy.load_state_dict(saved["policy"])
                    policy.eval()
                    controller = policy_controller(policy)
                else:
                    policy = _load_policy(saved, bank.config, args.device, bank.spot.dtype)
                    controller = sampled_controller(policy)
                metrics, tape = evaluate_controller(controller, bank, device=args.device, batch_size=1024,
                    mode_seed=seed + 920003, label=f"fresh/{case}/{seed}/{name}",
                    zeta=item["fixed_zeta"], progress=True)
                independent = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
                    bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
                values = tape["terminal_loss"].double().numpy()
                error = float(np.abs(independent["terminal_loss"] - values).max())
                losses[name] = values
                torch.save(tape, directory / f"{name}-tape.pt")
                seed_report["results"][name] = dict(checkpoint=str(path.resolve()), evaluation=metrics,
                    maximum_independent_cash_error=error)
                _report("counterfactual_fresh_result", case=case, seed=seed, method=name,
                    es95=metrics["es95"], maximum_independent_cash_error=error)
            seed_report["paired_comparisons"] = paired_intervals(losses, seed + 930003)
            (args.output / "confirmation.json").write_text(json.dumps(report, indent=2) + "\n")
            _report("counterfactual_fresh_seed_complete", case=case, seed=seed,
                    comparisons=seed_report["paired_comparisons"])


if __name__ == "__main__":
    main()
