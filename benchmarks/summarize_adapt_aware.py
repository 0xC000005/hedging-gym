"""Audit final adaptation tapes and report frozen comparisons without selection.

ES95 is recomputed independently with NumPy for each seed/market tape. Global
results average the four market ES values within each training seed, then
describe the three seed values. Losses from different policies or markets are
never pooled into an artificial tail distribution. No confidence intervals or
final-test choices are made by this script.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from benchmarks.compare_adapt_aware import (
    BUDGETS,
    FAMILIES,
    MODES,
    PATH_COUNTS,
    RATES,
    SEEDS,
    TEST_MARKETS,
)
from benchmarks.qualify_adaptation import load_bank
from benchmarks.summarize_fast_adaptation import es95
from hedging_gym.environment.finance import bank_subset, numpy_ledger


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def describe(values):
    """Descriptive training-seed spread, not an uncertainty interval."""
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(SEEDS),) and np.isfinite(values).all(), "invalid seed summary")
    return dict(by_seed=values.tolist(), mean=float(values.mean()),
                sample_std=float(values.std(ddof=1)), min=float(values.min()), max=float(values.max()))


def pretraining_work(output, development):
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for seed in SEEDS:
            path = (development/f"seed-{seed}"/"pretraining.json" if family == "original"
                    else output/"pretraining"/family/f"seed-{seed}"/"pretraining.json")
            metadata = read_json(path)
            initial = metadata.get("initial_pretraining", metadata)
            original_gradients = initial["expected_episode_rollouts"]
            original_initialization = sum(min(1024, count) for count in initial["source_paths"])
            additional = metadata.get("meta_pretraining")
            work = additional["work"] if additional else {}
            result[family][seed] = dict(source=str(path), source_unique_paths=sum(initial["source_paths"]),
                initial_training=dict(options=initial["options"],
                    gradient_episodes=original_gradients, initialization_forward_episodes=original_initialization,
                    total_forward_episodes=original_gradients+original_initialization,
                    training_seconds=initial["training_seconds"], total_seconds=initial["total_seconds"]),
                additional_training=({key: additional[key] for key in
                    ("recipe", "options", "work", "training_seconds", "objective", "gradient", "query_sampling")}
                    if additional else None),
                total_pretraining_forward_episodes=(original_gradients+original_initialization
                    +work.get("total_episode_rollouts", 0)),
                total_pretraining_gradient_episodes=original_gradients+work.get("gradient_episode_rollouts", 0),
                total_pretraining_seconds=metadata["total_seconds"])
    return result


def development_work(output):
    """Charge every completed rate-search curve, including unselected rates."""
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for seed in SEEDS:
            records = []
            for target in ("B", "C", "D"):
                for mode in MODES:
                    for rate in RATES:
                        path = (output/"develop"/family/f"seed-{seed}"/target/mode
                                /f"lr-{rate:g}"/"paths-4096"/"complete.json")
                        curve = read_json(path)
                        require(sorted(row["updates"] for row in curve["milestones"]) == list(BUDGETS),
                                f"incomplete development curve: {path}")
                        final = next(row for row in curve["milestones"] if row["updates"] == max(BUDGETS))
                        records.append(dict(
                            context_selection_forward_episodes=curve["selection"]["episode_rollouts"],
                            threshold_forward_episodes=curve["threshold_paths"],
                            gradient_episodes=final["gradient_episodes"],
                            adaptation_seconds=final["adaptation_seconds"],
                            selection_seconds=curve["selection"]["seconds"],
                            threshold_seconds=curve["threshold_seconds"],
                            evaluation_forward_episodes=sum(row["metrics"]["paths"] for row in curve["milestones"]),
                            evaluation_seconds=sum(row["metrics"]["evaluation_seconds"] for row in curve["milestones"])))
            totals = {key: sum(record[key] for record in records) for key in records[0]}
            totals["curves"] = len(records)
            totals["total_training_forward_episodes"] = sum(totals[key] for key in
                ("context_selection_forward_episodes", "threshold_forward_episodes", "gradient_episodes"))
            result[family][seed] = totals
    return result


def diagnosis_summary(output):
    """Fixed five-step development check; include work that the runner repeated."""
    targets, results, checked, maximum_error = ("B", "C", "D"), {}, 0, 0.
    identical_starts, baseline_checks = 0, 0
    baselines = {}
    for seed in SEEDS:
        for target in targets:
            directory = output/"develop"/"original"/f"seed-{seed}"/target/"finetune"/"lr-0.0003"/"paths-4096"
            curve = read_json(directory/"complete.json")
            tape = torch.load(directory/"0-tape.pt", map_location="cpu", weights_only=False)
            value = float(es95(tape["terminal_loss"].double().numpy()))
            recorded = next(row["metrics"]["es95"] for row in curve["milestones"] if row["updates"] == 0)
            require(np.isfinite(value) and value > 0 and abs(value-recorded) <= 1e-12,
                    f"diagnostic original-zero baseline does not reconcile: {directory}")
            baselines[seed, target] = value
            baseline_checks += 1
    for family in ("ordinary", "meta"):
        values, costs = {}, []
        for seed in SEEDS:
            for target in targets:
                directory = output/"diagnosis"/family/f"seed-{seed}"/target
                curve = read_json(directory/"complete.json")
                require((curve["family"], curve["policy_seed"], curve["target"], curve["mode"],
                         curve["rate"], curve["train_paths"]) == (family, seed, target, "finetune", 3e-4, 4096)
                        and sorted(row["updates"] for row in curve["milestones"]) == [0, 5],
                        f"diagnostic differs from the fixed five-step recipe: {directory}")
                rows = {row["updates"]: row for row in curve["milestones"]}
                for budget, row in rows.items():
                    tape = torch.load(directory/f"{budget}-tape.pt", map_location="cpu", weights_only=False)
                    loss = tape["terminal_loss"].double().numpy()
                    value = float(es95(loss))
                    error = abs(value-row["metrics"]["es95"])
                    require(loss.shape == (8192,) and np.isfinite(loss).all() and error <= 1e-12
                            and not bool(tape["constraint_violations"].any())
                            and row["gradient_episodes"] == 256*budget,
                            f"diagnostic tape does not reconcile: {directory}/{budget}")
                    if budget == 0:
                        prior = output/"develop"/family/f"seed-{seed}"/target/"finetune"/"lr-0.0003"/"paths-4096"
                        old = torch.load(prior/"0-tape.pt", map_location="cpu", weights_only=False)
                        require(tape.keys() == old.keys() and all(torch.equal(tape[key], old[key]) for key in tape),
                                f"diagnostic zero tape differs from development: {directory}")
                        identical_starts += 1
                    values[budget, seed, target] = value
                    maximum_error = max(maximum_error, error)
                    checked += 1
                costs.append(dict(context_selection_forward_episodes=curve["selection"]["episode_rollouts"],
                    threshold_forward_episodes=curve["threshold_paths"], gradient_episodes=rows[5]["gradient_episodes"],
                    zero_evaluation_forward_episodes=rows[0]["metrics"]["paths"],
                    five_update_evaluation_forward_episodes=rows[5]["metrics"]["paths"],
                    selection_seconds=curve["selection"]["seconds"], threshold_seconds=curve["threshold_seconds"],
                    adaptation_seconds=rows[5]["adaptation_seconds"],
                    zero_evaluation_seconds=rows[0]["metrics"]["evaluation_seconds"],
                    five_update_evaluation_seconds=rows[5]["metrics"]["evaluation_seconds"]))
        work = {key: sum(record[key] for record in costs) for key in costs[0]}
        work["curves"] = len(costs)
        work["total_training_forward_episodes"] = sum(work[key] for key in
            ("context_selection_forward_episodes", "threshold_forward_episodes", "gradient_episodes"))
        work["total_forward_episodes_including_evaluation"] = work["total_training_forward_episodes"]+sum(
            work[key] for key in ("zero_evaluation_forward_episodes", "five_update_evaluation_forward_episodes"))
        groups = {}
        for target in (*targets, "all_markets"):
            markets = targets if target == "all_markets" else (target,)
            groups[target] = {budget: dict(
                es95=describe([np.mean([values[budget, seed, market] for market in markets]) for seed in SEEDS]),
                ratio_to_original_zero=describe([np.mean([values[budget, seed, market]/baselines[seed, market]
                    for market in markets]) for seed in SEEDS])) for budget in (0, 5)}
        results[family] = dict(groups=groups, repeated_work=work)
    return dict(classification="Fixed five-step development diagnosis; not a final-test tuning candidate",
        recipe=dict(markets=list(targets), seeds=list(SEEDS), mode="finetune", rate=3e-4,
                    train_paths=4096, budgets=[0, 5], batch_size=256),
        aggregation="Normalize each seed/market ES by its original development zero-update ES, average markets within seed, then describe the three seed values. No confidence intervals.",
        work_scope="All context scoring, threshold initialization and zero-update evaluations were recomputed and are charged again; five-step training and evaluation are also charged.",
        families=results, audit=dict(passed=True, checked_diagnostic_tapes=checked,
            original_zero_baselines_reconciled=baseline_checks, identical_development_zero_tapes=identical_starts,
            maximum_saved_vs_numpy_es95_discrepancy=maximum_error, es95_absolute_tolerance=1e-12))


def summarize(output, development):
    selection = read_json(output/"selection.json")
    expected = len(FAMILIES)*len(SEEDS)*len(TEST_MARKETS)*len(MODES)*len(PATH_COUNTS)*len(BUDGETS)
    points, expected_paths, zero_losses = {}, set(), {}
    checked, maximum_discrepancy, zero_identity_checks = 0, 0., 0
    final_checkpoints, maximum_cash_error = 0, 0.
    cash_indices = np.linspace(0, 8191, 32, dtype=int)
    cash_banks = {target: bank_subset(load_bank(output/"banks"/f"{target}-eval.pt"),
                                    torch.from_numpy(cash_indices)) for target in TEST_MARKETS}
    final_rngs = {}
    for seed in SEEDS:
        for count in PATH_COUNTS:
            generator = torch.Generator().manual_seed(seed+102003)
            for _ in range(max(BUDGETS)):
                torch.randint(count, (256,), generator=generator)
            final_rngs[seed, count] = generator.get_state()
    started = time.perf_counter()
    print(f"Auditing {expected} final tapes; frozen rates read from selection.json", flush=True)
    for family in FAMILIES:
        for seed in SEEDS:
            for target in TEST_MARKETS:
                for mode in MODES:
                    rate = selection[family][mode]["rate"]
                    require(rate in RATES, f"undeclared selected rate: {family}/{mode}")
                    for count in PATH_COUNTS:
                        directory = (output/"test"/family/f"seed-{seed}"/target/mode
                                     /f"lr-{rate:g}"/f"paths-{count}")
                        curve = read_json(directory/"complete.json")
                        require((curve["family"], curve["policy_seed"], curve["target"], curve["mode"],
                                 curve["rate"], curve["train_paths"]) == (family, seed, target, mode, rate, count),
                                f"final curve differs from frozen recipe: {directory}")
                        require(sorted(row["updates"] for row in curve["milestones"]) == list(BUDGETS),
                                f"incomplete final curve: {directory}")
                        bank = cash_banks[target]
                        require(curve["config"] == json.loads(json.dumps(asdict(bank.config))),
                                f"curve/evaluation-bank financial configuration mismatch: {directory}")
                        for row in curve["milestones"]:
                            budget = row["updates"]
                            path = directory/f"{budget}-tape.pt"
                            expected_paths.add(path)
                            tape = torch.load(path, map_location="cpu", weights_only=False)
                            loss = tape["terminal_loss"].double().numpy()
                            require(loss.shape == (8192,) and row["metrics"]["paths"] == len(loss)
                                    and all(bool(torch.isfinite(value).all()) for value in tape.values()),
                                    f"nonfinite or malformed tape: {path}")
                            require(not bool(tape["constraint_violations"].any())
                                    and row["metrics"]["constraint_violations"] == 0,
                                    f"constraint violations: {path}")
                            positions = tape["positions"].numpy()
                            execution = curve["config"]["execution"]
                            require(np.all(positions >= np.asarray(execution["holding_lower"])-1e-10)
                                    and np.all(positions <= np.asarray(execution["holding_upper"])+1e-10),
                                    f"holdings outside configured bounds: {path}")
                            score = float(es95(loss))
                            discrepancy = abs(score-row["metrics"]["es95"])
                            maximum_discrepancy = max(maximum_discrepancy, discrepancy)
                            require(discrepancy <= 1e-12, f"saved ES95 disagrees with NumPy: {path}")
                            cash = numpy_ledger(bank.marks.numpy(), positions[cash_indices],
                                bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), bank.config)
                            cash_error = float(np.max(np.abs(cash["terminal_loss"]-loss[cash_indices])))
                            require(np.isfinite(cash_error) and cash_error <= 2e-6,
                                    f"independent NumPy cash reconstruction failed: {path}")
                            maximum_cash_error = max(maximum_cash_error, cash_error)
                            require(row["gradient_episodes"] == budget*256,
                                    f"target gradient-work mismatch: {path}")
                            if budget == 0:
                                key = family, seed, target
                                if key in zero_losses:
                                    require(np.array_equal(zero_losses[key], loss),
                                            f"zero-update policies differ across modes/counts: {path}")
                                    zero_identity_checks += 1
                                else:
                                    zero_losses[key] = loss.copy()
                            if budget == max(BUDGETS):
                                saved = torch.load(directory/f"{budget}-policy.pt", map_location="cpu",
                                                   weights_only=False)
                                state = saved["updater"]
                                require(saved["step"] == state["completed_steps"] == budget
                                        and state["pending_call"] is None and state["mode"] == mode
                                        and state["last_market"] == bank.config.market,
                                        f"final checkpoint adaptation state mismatch: {path}")
                                require(torch.equal(state["index_rng"], final_rngs[seed, count]),
                                        f"final checkpoint minibatch RNG mismatch: {path}")
                                options = state["options"]
                                require(options["seed"] == seed+2000 and options["batch_size"] == 256
                                        and options["learning_rate"] == rate
                                        and options["zeta_learning_rate"] == 3e-4,
                                        f"final checkpoint optimizer recipe mismatch: {path}")
                                require(bool(state["optimizer"]["state"]) and all(
                                    float(moment["step"]) == budget and all(
                                        bool(torch.isfinite(value).all()) for value in moment.values()
                                        if isinstance(value, torch.Tensor))
                                    for moment in state["optimizer"]["state"].values()),
                                    f"final Adam moments/step mismatch: {path}")
                                require(not state["requires_grad"]["source_embeddings"]
                                        and state["requires_grad"]["embedding"]
                                        and all(value == (mode == "finetune")
                                            for name, value in state["requires_grad"].items()
                                            if name.startswith("shared.")),
                                        f"final checkpoint adaptation parameter mismatch: {path}")
                                final_checkpoints += 1
                            points[family, mode, count, budget, target, seed] = dict(es95=score,
                                unique_target_paths=row["unique_target_paths"],
                                context_selection_forward_episodes=curve["selection"]["episode_rollouts"],
                                threshold_forward_episodes=curve["threshold_paths"],
                                gradient_episodes=row["gradient_episodes"],
                                adaptation_seconds=row["adaptation_seconds"],
                                total_target_training_seconds=(curve["selection"]["seconds"]
                                    +curve["threshold_seconds"]+row["adaptation_seconds"]))
                            checked += 1
                            if checked % 48 == 0 or checked == expected:
                                elapsed = time.perf_counter()-started
                                print(f"Tapes {checked}/{expected}; {elapsed:.1f}s; "
                                      f"ETA {elapsed*(expected-checked)/checked:.1f}s", flush=True)
    actual_paths = set((output/"test").rglob("*-tape.pt"))
    require(actual_paths == expected_paths and checked == expected,
            f"final tape inventory differs: {len(actual_paths)} present, {expected} expected")

    def values(family, mode, count, budget, target, field="es95"):
        markets = tuple(TEST_MARKETS) if target == "all_markets" else (target,)
        return np.asarray([np.mean([points[family, mode, count, budget, market, seed][field]
                                    for market in markets]) for seed in SEEDS])

    groups, comparisons = {}, {}
    for target in (*TEST_MARKETS, "all_markets"):
        groups[target], comparisons[target] = [], []
        for count in PATH_COUNTS:
            for budget in BUDGETS:
                for family in FAMILIES:
                    for mode in MODES:
                        current = values(family, mode, count, budget, target)
                        zero = values(family, mode, count, 0, target)
                        require(bool(np.all(zero > 0)), "percentage adaptation gains require positive baseline ES")
                        groups[target].append(dict(family=family, mode=mode,
                            rate=selection[family][mode]["rate"], train_paths=count, updates=budget,
                            es95=describe(current), own_zero_update_es95=describe(zero),
                            improvement_from_own_zero_percent=describe(100*(1-current/zero)),
                            work={field: describe(values(family, mode, count, budget, target, field))
                                  for field in ("unique_target_paths", "context_selection_forward_episodes",
                                      "threshold_forward_episodes", "gradient_episodes", "adaptation_seconds",
                                      "total_target_training_seconds")}))
                meta = values("meta", "finetune", count, budget, target)
                meta_zero = values("meta", "finetune", count, 0, target)
                for family, mode in (("original", "finetune"), ("original", "embedding"),
                                     ("ordinary", "finetune")):
                    reference = values(family, mode, count, budget, target)
                    reference_zero = values(family, mode, count, 0, target)
                    require(bool(np.all(reference > 0)), "relative comparison requires positive reference ES")
                    initialization = reference_zero-meta_zero
                    additional_adaptation = (meta_zero-meta)-(reference_zero-reference)
                    comparisons[target].append(dict(candidate="meta/finetune", reference=f"{family}/{mode}",
                        train_paths=count, updates=budget,
                        ratio_of_mean_es95=float(meta.mean()/reference.mean()),
                        es95_ratio=describe(meta/reference),
                        meta_improvement_percent=describe(100*(1-meta/reference)),
                        total_advantage_es95=describe(reference-meta),
                        zero_update_initialization_advantage_es95=describe(initialization),
                        extra_adaptation_advantage_es95=describe(additional_adaptation)))
    audit = dict(passed=True, expected_tapes=expected, checked_tapes=checked,
        curves=checked//len(BUDGETS), evaluation_paths_per_tape=8192,
        maximum_saved_vs_numpy_es95_discrepancy=maximum_discrepancy, es95_absolute_tolerance=1e-12,
        zero_update_identity_checks=zero_identity_checks,
        independent_cash_paths_per_tape=len(cash_indices), independent_cash_path_checks=checked*len(cash_indices),
        cash_path_indices=cash_indices.tolist(), maximum_cash_reconstruction_error=maximum_cash_error,
        cash_absolute_tolerance=2e-6, final_optimizer_rng_checkpoints=final_checkpoints,
        checks=["all expected final tapes and completed curves present; no extra final tapes",
            "frozen selected rates and family/seed/market/mode/path count",
            "finite tapes, zero violations and configured holding bounds",
            "independent NumPy fractional-tail ES95 reconciliation",
            "independent NumPy cash reconstruction on 32 evenly spaced evaluation paths per tape",
            "same-family zero-update losses identical across modes and path counts",
            "target gradient episode counts", "final adaptation mode, parameters, Adam steps/moments and replayed minibatch RNG"],
        exclusions="Source provenance and training/evaluation data-flow review are separate.",
        audit_seconds=time.perf_counter()-started)
    report = dict(classification="Three-training-seed held-out-market prototype comparison; descriptive evidence only",
        metric="Cost-inclusive terminal ES95 from saved test tapes; lower is better",
        seeds=list(SEEDS), test_markets=list(TEST_MARKETS), budgets=list(BUDGETS), train_path_counts=list(PATH_COUNTS),
        aggregation="ES within each policy/market; all_markets equally averages market ES within seed; then mean and sample standard deviation of the three seeds",
        uncertainty="Seed spread is descriptive. No confidence intervals or training-population significance claims.",
        selection="Pre-existing development selection.json is applied; no setting is selected from final test results.",
        frozen_selection=selection,
        comparison_decomposition="Positive favors meta: reference(update)-meta(update) = [reference(0)-meta(0)] + [(meta(0)-meta(update))-(reference(0)-reference(update))]",
        percentage_scope="Per-seed percentages are computed after market averaging for all_markets; ratio_of_mean_es95 is separately reported.",
        work_scope="Gradient episodes each include forward/backward work; counts are not FLOPs. Source banks are reused. Target unique paths include calibration. Initializations, context selection and all development rates are charged.",
        timing_scope="Saved wall times may overlap across jobs and are not uncontended latency measurements; bank generation and independent audits are separate.",
        pretraining_work=pretraining_work(output, development), development_work=development_work(output),
        diagnosis=diagnosis_summary(output) if (output/"diagnosis").exists() else None,
        groups=groups, meta_full_comparisons=comparisons, audit=audit)
    for filename, content in (("summary.json", report), ("audit.json", audit)):
        (output/filename).write_text(json.dumps(content, indent=2, allow_nan=False)+"\n")
    print(json.dumps(audit, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summarize(args.output, args.development)


if __name__ == "__main__":
    main()
