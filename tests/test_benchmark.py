"""Market/execution composition and fresh chronological evaluation boundaries."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from hedging_gym import benchmark, finance


def test_presets_share_observations_and_isolate_the_named_friction():
    basic = benchmark.benchmark_config()
    fields = finance.observation_fields(basic)
    expected = dict(operational_fixed="fixed_ticket", operational_minimum_fee="minimum_commission",
                   operational_minimum_trade="minimum_trade", operational_lots="trade_lot")
    for name, changed_field in expected.items():
        config = benchmark.benchmark_config(name)
        assert finance.observation_fields(config) == fields
        assert {key for key, value in asdict(config).items()
                if value != asdict(basic)[key]} == {changed_field}


def test_market_adaptation_and_execution_overlay_compose_without_cross_talk():
    for model in ("heston", "gbm", "bates"):
        basic = benchmark.benchmark_config(model=model)
        with_fee = benchmark.operational_config(basic, "operational_fixed")
        with_lots = benchmark.operational_config(with_fee, "operational_lots")
        assert with_lots.fixed_ticket == with_fee.fixed_ticket
        assert benchmark.operational_config(with_lots) == with_lots
        changed_fields = {"v0"} if model == "gbm" else {"theta", "v0"}
        for base in (basic, with_fee, with_lots):
            (_, a), (_, b), (_, returned) = benchmark.adaptation_configs(base)
            assert a == base == returned
            assert finance.observation_fields(a) == finance.observation_fields(b)
            assert {key for key, value in asdict(b).items()
                    if value != asdict(a)[key]} == changed_fields
        assert tuple((stage, benchmark.operational_config(config, "operational_fixed"))
                     for stage, config in benchmark.adaptation_configs(basic)) == benchmark.adaptation_configs(with_fee)
        no_fee = benchmark.operational_config(with_lots, fixed_ticket=(0., 0.))
        assert no_fee.fixed_ticket == (0., 0.) and no_fee.trade_lot == with_lots.trade_lot
    with pytest.raises(ValueError, match="stochastic-core"):
        benchmark.adaptation_configs(market_changes={"fixed_ticket": (.01, .01)})
    with pytest.raises(ValueError, match="stochastic-core"):
        benchmark.adaptation_configs(market_changes={"model": "gbm"})
    with pytest.raises(ValueError, match="fees and trading constraints"):
        benchmark.operational_config(basic, v0=.09)


def test_chronology_uses_fresh_eval_paths_and_persistent_update_state(monkeypatch):
    events, controller = [], SimpleNamespace(updates=0)
    def generate(config, paths, seed, **kwargs):
        events.append(("generate", seed))
        return SimpleNamespace(config=config, seed=seed)
    def update(bank):
        events.append(("update", bank.seed))
        controller.updates += 1
    def evaluate(current, bank, **kwargs):
        assert current is controller
        events.append(("evaluate", bank.seed))
        return dict(updates_seen=current.updates), dict(seed=bank.seed)
    monkeypatch.setattr(finance, "generate_market_bank", generate)
    monkeypatch.setattr(benchmark, "evaluate_controller", evaluate)
    report, tapes = benchmark.evaluate_adaptation(controller, train_paths=8,
        eval_paths=4, seed=100, updates_per_stage=2, update=update)
    assert events == [event for stage in range(3) for event in (
        ("generate", 101 + 5 * stage), ("evaluate", 101 + 5 * stage),
        ("generate", 100 + 5 * stage), ("update", 100 + 5 * stage),
        ("update", 100 + 5 * stage), ("generate", 103 + 5 * stage), ("evaluate", 103 + 5 * stage))]
    assert [stage["metrics"]["updates_seen"] for stage in report["stages"]] == [2, 4, 6]
    assert [stage["pre_update_metrics"]["updates_seen"] for stage in report["stages"]] == [0, 2, 4]
    assert [tapes[name]["seed"] for name in ("A", "B", "A_return")] == [103, 108, 113]
    events.clear()
    frozen, _ = benchmark.evaluate_adaptation(controller, train_paths=0,
        eval_paths=4, seed=200, updates_per_stage=0, update=update)
    assert events == [event for stage in range(3) for event in (
        ("generate", 201 + 5 * stage), ("evaluate", 201 + 5 * stage),
        ("generate", 203 + 5 * stage), ("evaluate", 203 + 5 * stage))]
    assert all(stage["training_paths_available"] == 0 for stage in frozen["stages"])
    assert controller.updates == 6
