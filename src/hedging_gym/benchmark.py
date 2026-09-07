"""ADAPTATION: independent market-regime and operational benchmark settings.

Presets are disclosed synthetic frictions, not an empirical fee calibration.
All share the one-call book and trading dates (35 observations for Heston).
Regime parameters change observably BETWEEN independent episodes; every stage
uses that identical observation schema and preserves the caller's learner state.
At S0=$100 and 1000 underlying units, .001 stock is one share, .1 call is one
100-share contract, and a .0001 money charge is $10 (money unit $100000).
"""
from dataclasses import asdict, replace
import time

import torch

from . import finance
from .evaluation import evaluate_controller


_OPERATIONAL_PRESETS = dict(basic={}, operational_fixed=dict(fixed_ticket=(.0001, .0001)),
    operational_minimum_fee=dict(minimum_commission=(.0001, .0001)),
    operational_minimum_trade=dict(minimum_trade=(.01, .1)),
    operational_lots=dict(trade_lot=(.001, .1)))
PRESET_NAMES = tuple(_OPERATIONAL_PRESETS)


def operational_config(base_config, name="basic", **execution_overrides):
    """Overlay only execution rules; unspecified fields are preserved.

    Named presets describe the common two-instrument book. Explicit overrides
    can compose frictions or set them to zero without changing market dynamics.
    ``basic`` is an empty overlay, not a reset of existing operational settings.
    """
    if name not in _OPERATIONAL_PRESETS:
        raise ValueError(f"unknown benchmark preset: {name}")
    if set(execution_overrides) - set(finance.EXECUTION_FIELDS):
        raise ValueError("operational overrides may change only fees and trading constraints")
    return replace(base_config, **dict(_OPERATIONAL_PRESETS[name], **execution_overrides))


def benchmark_config(name="basic", *, model="heston", **overrides):
    """Common market/book plus an optional, independent execution overlay.

    Preserves the original preset values. Explicit full-config overrides are
    for initial experiment construction, not changes within an A -> B -> A.
    """
    base = operational_config(finance.common_config(model=model), name)
    return replace(base, **dict(overrides, execution_features=True))


def adaptation_configs(base_config=None, *, market_changes=None):
    """Market-only A -> B -> A; execution, contracts and calendar stay fixed.

    Default B raises initial variance to .09 and, for Heston/Bates, long-run
    variance to .09. Explicit changes may alter only stochastic-core parameters
    within the selected model family. Apply operational_config to the baseline
    first to test this same sequence with a constant operational overlay.
    """
    baseline = benchmark_config() if base_config is None else base_config
    changes = dict(v0=.09)
    if baseline.model in ("heston", "bates"):
        changes["theta"] = .09
    if market_changes is not None:
        changes = dict(market_changes)
    if set(changes) - {"v0", *finance.market_parameter_names(baseline)}:
        raise ValueError("adaptation may change only stochastic-core parameters, not execution or book terms")
    changed = replace(baseline, **changes)
    if changed == baseline:
        raise ValueError("regime B must change a stochastic-core parameter")
    return (("A", baseline), ("B", changed), ("A_return", baseline))


def evaluate_adaptation(controller, *, train_paths, eval_paths, seed,
                        base_config=None, market_changes=None, updates_per_stage=0, update=None,
                        simulation_substeps=1,
                        device="cpu", dtype=torch.float32, batch_size=1024,
                        progress=False):
    """Evaluate one persistent controller chronologically, without creating a learner.

    Each stage evaluates before and after exactly updates_per_stage calls to
    ``update(training_bank)``; only the current training bank reaches the callback.
    Controller/callback closures retain state across stages. Future-stage banks
    do not exist yet. Pre/post evaluations use separately seeded fresh paths.
    Return-A BEFORE updating tests forgetting relative to original A AFTER
    updating; return-A after updating instead tests recovery. Evaluation itself
    must not update learner weights.
    The caller supplies config-aware control; legacy bound actors are not rebound.
    Zero updates generates no training bank. Callback calls are charged, not
    misreported as optimizer steps or inferred numbers of consumed paths.

    Returns (report, tapes), using the common evaluator's terminal ES. Existing
    stage keys/metrics hold post-update results; ``A_before`` etc. retain pre-update
    tapes. Comparisons use independent paths, not a paired-path error estimate.
    This is a data/execution contract, not a trained-method comparison.
    """
    if (not isinstance(updates_per_stage, int) or updates_per_stage < 0
            or eval_paths < 1 or (updates_per_stage and train_paths < 1)):
        raise ValueError("nonnegative update budget and positive used path counts required")
    if updates_per_stage and update is None:
        raise ValueError("positive update budget requires update(training_bank)")
    device = torch.device(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    configs = adaptation_configs(base_config, market_changes=market_changes)
    report = dict(classification="ADAPTATION / BENCHMARK HARNESS", model=configs[0][1].model,
        seed=seed, updates_per_stage=updates_per_stage,
        simulation_substeps=simulation_substeps, stages=[])
    tapes = {}
    for index, (stage, config) in enumerate(configs):
        train_seed, before_seed, before_mode_seed, eval_seed, mode_seed = (
            seed+5*index+j for j in range(5))
        if progress:
            print(f"{report['model']}/{stage}: train={train_paths if updates_per_stage else 0}, "
                  f"eval={eval_paths} before+after, callback calls={updates_per_stage}, "
                  f"seeds train/pre/pre-mode/post/post-mode="
                  f"{train_seed}/{before_seed}/{before_mode_seed}/{eval_seed}/{mode_seed}", flush=True)
        synchronize()
        started = time.perf_counter()
        before_bank = finance.generate_market_bank(config, eval_paths, before_seed,
            device=device, dtype=dtype, substeps=simulation_substeps)
        synchronize()
        before_bank_seconds = time.perf_counter()-started
        before_metrics, tapes[stage+"_before"] = evaluate_controller(controller, before_bank,
            device=device, batch_size=batch_size, mode_seed=before_mode_seed,
            label=f"ADAPTATION / {report['model']}/{stage}/before", progress=progress)
        del before_bank
        train_seconds, update_seconds = 0., 0.
        if updates_per_stage:
            synchronize()
            started = time.perf_counter()
            training_bank = finance.generate_market_bank(config, train_paths, train_seed,
                device=device, dtype=dtype, substeps=simulation_substeps)
            synchronize()
            train_seconds = time.perf_counter()-started
            started = time.perf_counter()
            for _ in range(updates_per_stage):
                update(training_bank)
            synchronize()
            update_seconds = time.perf_counter()-started
            del training_bank
        synchronize()
        started = time.perf_counter()
        evaluation_bank = finance.generate_market_bank(config, eval_paths, eval_seed,
            device=device, dtype=dtype, substeps=simulation_substeps)
        synchronize()
        eval_bank_seconds = time.perf_counter()-started
        metrics, tapes[stage] = evaluate_controller(controller, evaluation_bank,
            device=device, batch_size=batch_size, mode_seed=mode_seed,
            label=f"ADAPTATION / {report['model']}/{stage}", progress=progress)
        report["stages"].append(dict(stage=stage, config=asdict(config),
            pre_update_metrics=before_metrics, pre_update_evaluation_seed=before_seed,
            pre_update_mode_seed=before_mode_seed, pre_update_bank_seconds=before_bank_seconds,
            training_paths_available=train_paths if updates_per_stage else 0,
            evaluation_paths=eval_paths, training_seed=train_seed if updates_per_stage else None,
            evaluation_seed=eval_seed, mode_seed=mode_seed,
            update_callback_calls=updates_per_stage, training_bank_seconds=train_seconds,
            update_seconds=update_seconds, evaluation_bank_seconds=eval_bank_seconds,
            metrics=metrics))
        del evaluation_bank
    return report, tapes
