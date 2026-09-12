"""Audit matched adaptation checkpoints and summarize their common-path losses."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.summarize_fast_adaptation import SEEDS, es95, interval
from experiments.qualify_adaptation import load_bank
from hedging_gym.config import config_from_dict
from hedging_gym.finance import bank_subset, numpy_ledger


MODES, BUDGETS = ("embedding", "finetune"), (0, 10, 50, 200)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, (float, int)) or bool(np.isfinite(value))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def optimizer_recipe(state):
    # Parameter sets and learned moments differ; the Adam recipe must match.
    return [{k: v for k, v in group.items() if k != "params"}
            for group in state["optimizer"]["param_groups"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(960002)
    report = dict(classification="Development update-capacity diagnostic; three policy seeds and one learning-rate recipe; no optimized-method claims",
        metric="Cost-inclusive terminal ES95 pooled across evaluation paths within each seed",
        aggregation="Arithmetic mean of the three per-policy ES values, not ES of losses pooled across policies",
        seeds=list(SEEDS), budgets=list(BUDGETS), difference="finetune minus embedding; negative favors full updates",
        bootstrap="Paired paths shared across seeds, with policy-seed blocks; exploratory with three seeds",
        bootstrap_seed=960002, bootstrap_repeats=500,
        cost_scope="Context selection and threshold initialization charged to both arms; evaluation excluded",
        timing_scope="Saved wall times are diagnostic only; concurrent jobs may overlap",
        work_scope="Episode counts are not FLOPs; gradient episodes include a forward and backward pass",
        bank_paths=dict(train=4096, calibration=1024, evaluation=8192), shared_pretraining={}, targets={})
    for seed in SEEDS:
        meta = json.loads((args.output/f"seed-{seed}"/"pretraining.json").read_text())
        report["shared_pretraining"][seed] = {k: meta[k] for k in (
            "source_paths", "expected_episode_rollouts", "initialization_seconds",
            "training_seconds", "total_seconds", "options")}
    checked, maximum_cash_error = 0, 0.
    audit_indices = np.linspace(0, 8191, 32, dtype=int)
    for target in ("A", "B", "C", "D"):
        bank = bank_subset(load_bank(args.output/"banks"/f"{target}-eval.pt"), torch.from_numpy(audit_indices))
        losses = {(mode, budget): [] for mode in MODES for budget in BUDGETS}
        result = {mode: {budget: dict(es95_by_seed=[], work_by_seed=[], timing_seconds_by_seed=[])
                         for budget in BUDGETS} for mode in MODES}
        for seed in SEEDS:
            directory = args.output/f"seed-{seed}"/target
            curves = {mode: json.loads((directory/mode/"curve.json").read_text()) for mode in MODES}
            initial = json.loads((directory/"initialization.json").read_text())
            baseline, zero_loss = {}, {}
            generator = torch.Generator().manual_seed(seed+101003)
            seen = torch.zeros(4096, dtype=torch.bool)
            seen[:1024] = True
            previous_budget = 0
            for budget in BUDGETS:
                for _ in range(budget-previous_budget):
                    seen[torch.randint(4096, (256,), generator=generator)] = True
                previous_budget = budget
                states = {}
                for mode, curve in curves.items():
                    label = f"{target}/seed-{seed}/{mode}/{budget}"
                    require(finite(curve) and curve["initialization"] == initial, f"initialization: {label}")
                    require(curve["policy_seed"] == seed and curve["target"] == target
                            and curve["mode"] == mode and curve["batch_size"] == 256
                            and curve["minibatch_seed"] == seed+101003
                            and config_from_dict(curve["config"]) == bank.config, f"recipe: {label}")
                    require(sorted(r["updates"] for r in curve["milestones"]) == list(BUDGETS), f"milestones: {label}")
                    row = next(r for r in curve["milestones"] if r["updates"] == budget)
                    tape = load(directory/mode/f"{budget}-tape.pt")
                    saved = load(directory/mode/f"{budget}-policy.pt")
                    state = states[mode] = saved["updater"]
                    loss = tape["terminal_loss"].double().numpy()
                    require(finite(tape) and loss.shape == (8192,) and finite(state["policy"])
                            and finite(state["optimizer"]) and finite(state["zeta"]), f"nonfinite/shape: {label}")
                    require(not tape["constraint_violations"].any() and row["metrics"]["constraint_violations"] == 0,
                            f"constraint violations: {label}")
                    positions, execution = tape["positions"].numpy(), curve["config"]["execution"]
                    require(np.all(positions >= np.asarray(execution["holding_lower"])-1e-10)
                            and np.all(positions <= np.asarray(execution["holding_upper"])+1e-10), f"holdings: {label}")
                    require(abs(es95(loss)-row["metrics"]["es95"]) <= 1e-12, f"ES reconciliation: {label}")
                    cash = numpy_ledger(bank.marks.numpy(), positions[audit_indices],
                        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
                    cash_error = float(np.abs(cash["terminal_loss"]-loss[audit_indices]).max())
                    require(finite(cash) and cash_error <= 2e-6, f"cash reconstruction: {label}")
                    maximum_cash_error = max(maximum_cash_error, cash_error)
                    require(saved["step"] == state["completed_steps"] == budget and state["pending_call"] is None
                            and saved["report"]["milestones"][-1] == row, f"checkpoint step: {label}")
                    require(torch.equal(state["index_rng"], generator.get_state()), f"minibatch RNG: {label}")
                    moments = state["optimizer"]["state"]
                    require((not moments if budget == 0 else bool(moments))
                            and all(float(v["step"]) == budget for v in moments.values()), f"Adam steps: {label}")
                    if budget == 0:
                        baseline[mode], zero_loss[mode] = state["policy"], loss.copy()
                    changed = [name.removeprefix("shared.") for name, value in state["policy"].items()
                               if name.startswith("shared.") and not torch.equal(value, baseline[mode][name])]
                    require(sorted(changed) == sorted(row["shared_parameters_changed"])
                            and bool(changed) == (mode == "finetune" and budget > 0), f"shared weights: {label}")
                    require(torch.equal(state["policy"]["source_embeddings"], baseline[mode]["source_embeddings"])
                            and not state["requires_grad"]["source_embeddings"], f"source contexts: {label}")
                    require(initial["calibration_distinct_paths"] == initial["threshold_initialization_paths"] == 1024
                            and row["unique_target_paths"] == 1024+int(seen.sum())
                            and row["gradient_episode_rollouts"] == budget*256
                            and row["calibration_episode_rollouts"] == initial["episode_rollouts"] == len(initial["scores"])*1024,
                            f"work accounting: {label}")
                    point = result[mode][budget]
                    point["es95_by_seed"].append(float(es95(loss)))
                    point["work_by_seed"].append(dict(unique_target_paths=row["unique_target_paths"],
                        context_selection_forward_episodes=initial["episode_rollouts"], threshold_forward_episodes=1024,
                        gradient_episodes=budget*256, total_forward_episodes=initial["episode_rollouts"]+1024+budget*256,
                        active_policy_parameters=curve["active_adaptation_parameters"], threshold_parameters=1))
                    point["timing_seconds_by_seed"].append(dict(context_selection=initial["seconds"],
                        threshold_initialization=initial["threshold_initialization_seconds"], updates=row["adaptation_seconds"],
                        adaptation_total=initial["seconds"]+initial["threshold_initialization_seconds"]+row["adaptation_seconds"],
                        evaluation_excluded=row["metrics"]["evaluation_seconds"]))
                    losses[mode, budget].append(loss)
                    checked += 1
                left, right = states.values()
                require(left["options"] == right["options"] and optimizer_recipe(left) == optimizer_recipe(right),
                        f"optimizer recipe mismatch: {target}/{seed}/{budget}")
                if budget == 0:
                    require(np.array_equal(zero_loss["embedding"], zero_loss["finetune"])
                            and torch.equal(left["zeta"], right["zeta"])
                            and all(torch.equal(value, right["policy"][name]) if isinstance(value, torch.Tensor)
                                    else value == right["policy"][name] for name, value in left["policy"].items()),
                            f"zero-update identity: {target}/{seed}")
        result["paired_difference"] = {}
        for budget in BUDGETS:
            for mode in MODES:
                point = result[mode][budget]
                point["es95_mean"] = float(np.mean(point["es95_by_seed"]))
            differences = np.asarray(result["finetune"][budget]["es95_by_seed"])-result["embedding"][budget]["es95_by_seed"]
            point = result["paired_difference"][budget] = dict(by_seed=differences.tolist(), mean=float(differences.mean()))
            if budget == 200:
                point["exploratory_ci95"] = interval(np.stack(losses["finetune", budget]),
                    np.stack(losses["embedding", budget]), rng, 500)
            print(target, budget, point, flush=True)
        report["targets"][target] = result
    report["audit"] = dict(passed=True, tapes_and_checkpoints=checked,
        independent_cash_paths_per_tape=32, cash_path_indices=audit_indices.tolist(),
        maximum_cash_reconstruction_error=maximum_cash_error, cash_absolute_tolerance=2e-6,
        checks="Finite losses, tapes and optimizer states; ES reconciliation; bounds; identical starts; shared weights; Adam recipes/steps; replayed minibatch RNG and unique paths")
    (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
