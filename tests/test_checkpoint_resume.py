"""An interrupted learner must continue the same training trajectory."""
from dataclasses import asdict, replace

import pytest
import torch

from hedging_gym import benchmark_config
from hedging_gym.config import TimeGrid
from hedging_gym.finance import generate_market_bank
from methods.training import train_policy
from methods.hybrid import train_hybrid


def test_legacy_qe_checkpoint_and_bank_cannot_resume_as_qe_m(tmp_path):
    from experiments.baselines import _load_bank
    from methods.checkpoints import load_checkpoint

    modern = benchmark_config()
    legacy = replace(modern, market=replace(modern.market, scheme="qe"))
    saved_config = asdict(legacy)
    saved_config["market"].pop("scheme")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(method="dh", config=saved_config), checkpoint)
    assert load_checkpoint(checkpoint, method="dh", config=legacy)["method"] == "dh"
    with pytest.raises(ValueError, match="configuration differs"):
        load_checkpoint(checkpoint, method="dh", config=modern)

    bank = tmp_path / "bank.pt"
    torch.save(dict(config=saved_config, seed=19,
        spot=torch.ones(2, 31), variance=torch.full((2, 31), .04),
        marks=torch.ones(2, 31, legacy.n_assets), liability=torch.zeros(2, 31)), bank)
    assert _load_bank(bank, legacy, 19, "cpu").config.market.scheme == "qe"
    with pytest.raises(ValueError, match="configuration/seed differs"):
        _load_bank(bank, modern, 19, "cpu")


def test_legacy_pickled_market_uses_instance_fields_and_preserves_contract():
    from copy import deepcopy
    from experiments.compare_fast_adaptation import SOURCE_MARKETS, market_config
    from methods.checkpoints import saved_config, saved_market

    config = market_config(SOURCE_MARKETS[0])
    old = deepcopy(config)
    vars(old.market).pop("scheme")
    vars(old).pop("settlement")
    assert old.market.scheme == "qe_m"  # Today's class fallback is not saved evidence.
    assert saved_market(old.market) == config.market
    assert saved_config(old) == config
    assert saved_market(asdict(config.market)) == config.market
    assert saved_market(None) is None
    assert config.market.scheme == "qe" and config.market.kappa == 3.
    assert config.risk.objective == "es" and config.n_assets == 2
    assert config.time_grid.days_per_year == 252 and config.n_decisions == 30


@pytest.mark.parametrize("method", ["dh", "ntb", "hpo"])
def test_split_training_matches_uninterrupted(method, tmp_path):
    bank = generate_market_bank(benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3)),
                                24, 440, dtype=torch.float64)
    def train(**kwargs):
        return (train_hybrid(bank, **kwargs) if method == "hpo"
                else train_policy(method, bank, **kwargs))
    options = dict(seed=7, batch_size=8, hidden=(8,), progress=False)
    full, full_meta = train(updates=4, **options)
    path = tmp_path / "latest.pt"
    train(updates=2, checkpoint_path=path, checkpoint_every=1, **options)
    # Older snapshots omit newly added fields whose defaults preserve the task.
    legacy = torch.load(path, weights_only=False)
    for key in ("step_days", "trade_at_maturity"):
        legacy["config"]["time_grid"].pop(key)
    legacy["config"].pop("settlement")
    torch.save(legacy, path)
    resumed, meta = train(updates=4, resume_from=path, checkpoint_path=path, **options)
    for name, value in full.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        else:
            assert value == resumed.state_dict()[name]
    assert meta["zeta"] == full_meta["zeta"]
    assert (tmp_path / "latest-early.pt").exists()
    saved = torch.load(path, weights_only=False)
    assert saved["step"] == 4 and saved["optimizer"]["state"]
    with pytest.raises(ValueError, match="precede"):
        train(updates=1, resume_from=path, **options)
