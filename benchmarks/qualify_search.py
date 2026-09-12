"""Evaluate saved hybrid weights and common-continuation search ablations.

All search variants use identical feedback, paths and base candidate budgets.
Gradient refinement adds evaluations and is charged separately. Small subsets
are development diagnostics, not evidence of a statistically resolved winner.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.qualify_policies import load_bank
from hedging_gym.baselines._shared.controllers import policy_controller
from hedging_gym.baselines._shared.training import _report
from hedging_gym.baselines.cem import RolloutPlanner
from hedging_gym.baselines.hpo import HybridPolicy
from hedging_gym.environment.finance import bank_subset, numpy_ledger
from hedging_gym.evaluation import evaluate_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", default="basic")
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", choices=("hpo_greedy", "hpo_sampled", "unguided", "guided", "refined"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    bank = bank_subset(load_bank(args.run_dir / "development_bank.pt", args.preset), slice(0, args.paths))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = HybridPolicy(bank.config, hidden=saved["options"]["hidden"]).to(args.device)
    policy.load_state_dict(saved["policy"])
    policy.eval()
    original = {name: value.detach().clone() for name, value in policy.named_parameters()}
    zeta = float(saved["zeta"])

    def sampled(observed, ledger, time_index, config):
        return policy(observed, ledger.positions, config.execution.holding_lower,
                      config.execution.holding_upper, deterministic=False).target_holdings
    sampled.action_selection = "sampled hybrid modes"
    controllers = dict(hpo_greedy=policy_controller(policy), hpo_sampled=sampled)
    for name, guided, gradients in (("unguided", False, 0), ("guided", True, 0), ("refined", True, 3)):
        controllers[name] = RolloutPlanner(policy, zeta=zeta, candidates=args.candidates,
            scenarios=args.scenarios, iterations=2, seed=200010, guided=guided,
            gradient_steps=gradients, progress=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(scope="Common-continuation development diagnostic; not a final comparison",
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        results={})
    for name, controller in controllers.items():
        if args.variants and name not in args.variants:
            continue
        metrics, tape = evaluate_controller(controller, bank, device=args.device,
            batch_size=args.batch_size, label=name, zeta=zeta, progress=True)
        reference = numpy_ledger(bank.marks.numpy(), tape["positions"].numpy(),
            bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
        discrepancy = float(np.max(np.abs(reference["terminal_loss"] - tape["terminal_loss"].numpy())))
        np.testing.assert_allclose(reference["terminal_loss"], tape["terminal_loss"], atol=2e-6, rtol=0)
        record = dict(evaluation=metrics, maximum_cash_reconstruction_error=discrepancy)
        if isinstance(controller, RolloutPlanner):
            record["planning"] = controller.metadata()
        torch.save(tape, args.output_dir / f"{name}-tape.pt")
        report["results"][name] = record
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
        _report("search_qualification", method=name, es95=metrics["es95"], **record)
    for name, value in policy.named_parameters():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


if __name__ == "__main__":
    main()
