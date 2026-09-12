"""Market/execution composition and fresh chronological evaluation boundaries."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from hedging_gym import benchmark, finance
from hedging_gym.config import EuropeanOption, ExecutionConfig, GBMConfig, PortfolioConfig, RiskConfig, TimeGrid


def test_presets_share_observations_and_isolate_the_named_friction():
    basic = benchmark.benchmark_config()
    fields = finance.observation_fields(basic)
    expected = dict(operational_fixed="fixed_ticket", operational_minimum_fee="minimum_commission",
                   operational_minimum_trade="minimum_trade", operational_lots="trade_lot")
    for name, changed_field in expected.items():
        config = benchmark.benchmark_config(name=name)
        assert finance.observation_fields(config) == fields
        assert config.market == basic.market and config.portfolio == basic.portfolio
        assert config.time_grid == basic.time_grid and config.risk == basic.risk
        assert {key for key, value in asdict(config.execution).items()
                if value != asdict(basic.execution)[key]} == {changed_field}


def test_explicit_execution_replaces_preset_defaults():
    for fee in (0., .02):
        execution = ExecutionConfig(fixed_ticket=fee)
        config = benchmark.benchmark_config(name="operational_fixed", execution=execution)
        assert config.execution is execution
        assert config.execution.fixed_ticket == fee
        overlaid = benchmark.operational_config(config, "operational_fixed")
        assert overlaid.execution.fixed_ticket == .0001
        assert benchmark.operational_config(config, "operational_fixed", fixed_ticket=fee).execution == execution


def test_market_adaptation_and_execution_overlay_compose_without_cross_talk():
    for model in ("heston", "gbm", "bates"):
        basic = benchmark.benchmark_config(model=model, risk=RiskConfig(alpha=.99))
        with_fee = benchmark.operational_config(basic, "operational_fixed")
        with_lots = benchmark.operational_config(with_fee, "operational_lots")
        assert with_lots.execution.fixed_ticket == with_fee.execution.fixed_ticket
        assert benchmark.operational_config(with_lots) == with_lots
        changed_fields = {"v0"} if model == "gbm" else {"theta", "v0"}
        for base in (basic, with_fee, with_lots):
            (_, a), (_, b), (_, returned) = benchmark.adaptation_configs(base)
            assert a == base == returned
            assert b.market.v0 == pytest.approx(2.25 * a.market.v0)
            assert finance.observation_fields(a) == finance.observation_fields(b)
            assert {key for key, value in asdict(b.market).items()
                    if value != asdict(a.market)[key]} == changed_fields
            assert b.execution == a.execution and b.portfolio == a.portfolio
            assert b.time_grid == a.time_grid and b.risk == a.risk
        assert tuple((stage, benchmark.operational_config(config, "operational_fixed"))
                     for stage, config in benchmark.adaptation_configs(basic)) == benchmark.adaptation_configs(with_fee)
        no_fee = benchmark.operational_config(with_lots, fixed_ticket=0.)
        assert no_fee.execution.fixed_ticket == 0.
        assert no_fee.execution.trade_lot == with_lots.execution.trade_lot
    with pytest.raises(ValueError, match="stochastic-core"):
        benchmark.adaptation_configs(market_changes={"fixed_ticket": (.01, .01)})
    with pytest.raises(ValueError, match="stochastic-core"):
        benchmark.adaptation_configs(market_changes={"model": "gbm"})
    with pytest.raises(ValueError, match="stochastic-core"):
        benchmark.adaptation_configs(market_changes={"spot0": 2.})
    with pytest.raises(ValueError, match="fees and trading constraints"):
        benchmark.operational_config(basic, v0=.09)


def test_benchmark_defaults_follow_the_market_calendar_and_selected_book():
    grid = TimeGrid(n_steps=90, days_per_year=365)
    market = GBMConfig(spot0=100.)
    default = benchmark.benchmark_config(model=market, time_grid=grid)
    assert default.market is market and default.time_grid is grid
    assert default.portfolio.liability == EuropeanOption(100., grid.horizon)
    assert default.portfolio.hedges == (EuropeanOption(100., 2*grid.horizon),)
    for hedges in ((), (EuropeanOption(90., 2*grid.horizon, "put"),
                        EuropeanOption(100., 2*grid.horizon),
                        EuropeanOption(110., 3*grid.horizon))):
        book = PortfolioConfig(default.portfolio.liability, hedges)
        config = benchmark.benchmark_config(model=market, time_grid=grid, portfolio=book)
        assert config.portfolio is book
        assert config.execution.vector("proportional", config.n_assets) == (.0005,) + (.01,) * len(hedges)
        fixed = benchmark.operational_config(config, "operational_fixed")
        assert fixed.execution.vector("fixed_ticket", config.n_assets) == (.0001,) * config.n_assets
        lots = benchmark.operational_config(fixed, "operational_lots")
        assert lots.execution.vector("trade_lot", config.n_assets) == (.001,) + (.1,) * len(hedges)
        assert finance.observation_fields(lots) == finance.observation_fields(config)


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
