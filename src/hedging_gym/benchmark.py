"""Independent market, portfolio, execution and risk benchmark settings.

Presets are disclosed synthetic frictions, not an empirical fee calibration.
The default book has one hedge call. Regime parameters change observably between
independent episodes; each stage preserves the observation schema and learner state.
At S0=$100 and 1000 underlying units, .001 stock is one share, .1 call is one
100-share contract, and a .0001 money charge is $10 (money unit $100000).
"""
from dataclasses import asdict, fields, replace
import time

import torch

from . import finance
from .config import (
    BatesConfig, EuropeanOption, ExecutionConfig, GBMConfig, HedgingConfig,
    HestonConfig, PortfolioConfig, RiskConfig, TimeGrid,
)
from .evaluation import evaluate_controller


PRESET_NAMES = ("basic", "operational_fixed", "operational_minimum_fee",
                "operational_minimum_trade", "operational_lots")


def operational_config(base_config, name="basic", **execution_overrides):
    """Overlay only execution rules; unspecified fields are preserved.

    Amount presets broadcast to the selected book; order sizes distinguish stock
    from option contracts. Explicit overrides
    can compose frictions or set them to zero without changing market dynamics.
    ``basic`` is an empty overlay, not a reset of existing operational settings.
    """
    if name not in PRESET_NAMES:
        raise ValueError(f"unknown benchmark preset: {name}")
    if set(execution_overrides) - {field.name for field in fields(ExecutionConfig)}:
        raise ValueError("operational overrides may change only fees and trading constraints")
    presets = dict(basic={}, operational_fixed=dict(fixed_ticket=.0001),
        operational_minimum_fee=dict(minimum_commission=.0001),
        operational_minimum_trade=dict(minimum_trade=(.01,) + (.1,) * (base_config.n_assets-1)),
        operational_lots=dict(trade_lot=(.001,) + (.1,) * (base_config.n_assets-1)))
    if not presets[name] and not execution_overrides:
        return base_config
    execution = replace(base_config.execution, **dict(presets[name], **execution_overrides))
    return replace(base_config, execution=execution)


def benchmark_config(*, model="heston", time_grid=None, portfolio=None,
                     execution=None, risk=None, name="basic"):
    """Construct a benchmark from independent configuration components.

    Defaults sell one ATM call at the grid horizon and hedge with stock and one
    ATM call at twice that horizon. Default fees and holding bounds follow the
    selected instrument count. A named preset supplies execution defaults;
    explicit execution replaces that complete component, including zero fees.
    """
    models = {"gbm": GBMConfig, "heston": HestonConfig, "bates": BatesConfig}
    if isinstance(model, str):
        if model not in models:
            raise ValueError(f"unknown market model: {model}")
        market = models[model]()
    else:
        market = model
    grid = TimeGrid() if time_grid is None else time_grid
    book = (PortfolioConfig(
        liability=EuropeanOption(strike=market.spot0, maturity=grid.horizon),
        hedges=(EuropeanOption(strike=market.spot0, maturity=2*grid.horizon),))
        if portfolio is None else portfolio)
    rules = ExecutionConfig(proportional=(.0005,) + (.01,) * (book.n_assets-1),
        holding_lower=(-1.,) * book.n_assets,
        holding_upper=(2.,) + (1.,) * (book.n_assets-1))
    base = HedgingConfig(market=market, time_grid=grid, portfolio=book,
        execution=rules, risk=RiskConfig() if risk is None else risk)
    preset = operational_config(base, name)
    return preset if execution is None else replace(preset, execution=execution)


def adaptation_configs(base_config=None, *, market_changes=None):
    """Market-only A -> B -> A; execution, contracts and calendar stay fixed.

    Default B scales initial volatility by 1.5 and, for Heston/Bates, long-run
    volatility likewise (variances by 2.25). Explicit changes may alter only
    stochastic-core parameters within the selected model family. Apply
    operational_config first to test this sequence with a constant execution overlay.
    """
    baseline = benchmark_config() if base_config is None else base_config
    changes = dict(v0=2.25 * baseline.market.v0)
    if baseline.market.model in ("heston", "bates"):
        changes["theta"] = 2.25 * baseline.market.theta
    if market_changes is not None:
        changes = dict(market_changes)
    if set(changes) - {"v0", *finance.market_parameter_names(baseline.market)}:
        raise ValueError("adaptation may change only stochastic-core parameters, not execution or book terms")
    changed = replace(baseline, market=replace(baseline.market, **changes))
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
    The caller supplies config-aware control with the selected observation schema.
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
    report = dict(classification="Market adaptation benchmark", model=configs[0][1].market.model,
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
            label=f"{report['model']}/{stage}/before", progress=progress)
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
            label=f"{report['model']}/{stage}", progress=progress)
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
