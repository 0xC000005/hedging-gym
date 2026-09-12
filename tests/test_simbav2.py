"""Optional official-donor tests; SIMBAV2_SOURCE selects the pinned checkout."""
import copy
from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from hedging_gym import benchmark_config
from hedging_gym.finance import generate_market_bank
from hedging_gym.config import RiskConfig, TimeGrid
from methods.simbav2 import RawLossReplay


def test_simba_rejects_unsupported_contract_before_loading_donor():
    from methods.simbav2 import SimBaV2Hedger
    config = benchmark_config(model="gbm")
    with pytest.raises(ValueError, match="terminal ES only"):
        SimBaV2Hedger(replace(config, risk=RiskConfig(objective="mse")), "unused")
    with pytest.raises(ValueError, match="finite holding bounds"):
        SimBaV2Hedger(replace(config, execution=replace(config.execution, holding_lower=None)), "unused")


def test_replay_relabels_terminal_rewards_and_roundtrips_rng():
    replay = RawLossReplay(3, 7)
    observed = np.arange(8, dtype=np.float32).reshape(4, 2)
    replay.add(observed, np.zeros((4, 1)), observed+1, [False, True, False, True], [0., 2., 0., 4.])
    saved = replay.state_dict()
    old = replay.sample(32, zeta=1., alpha=.5)
    replay.load_state_dict(saved)
    new = replay.sample(32, zeta=2.5, alpha=.5)
    np.testing.assert_array_equal(old["observation"], new["observation"])
    terminal = new["terminated"]
    assert np.all(new["reward"][~terminal] == 0)
    assert np.all(new["reward"][terminal] != old["reward"][terminal])
    clone = RawLossReplay(1, 99)
    clone.load_state_dict(replay.state_dict())
    a, b = replay.sample(32, zeta=2., alpha=.95), clone.sample(32, zeta=2., alpha=.95)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.skipif(not os.environ.get("SIMBAV2_SOURCE"), reason="requires pinned external SimBaV2 runtime")
def test_official_update_full_checkpoint_continuation_and_ledger(tmp_path):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from methods.simbav2 import SimBaV2Hedger, NETWORKS
    from experiments.qualify_simbav2 import collect, evaluate
    import jax
    config = benchmark_config(model="gbm", time_grid=TimeGrid(n_steps=3))
    bank = generate_market_bank(config, 8, 19)
    kwargs = dict(donor_path=os.environ["SIMBAV2_SOURCE"], seed=7, total_updates=4,
                  actor_width=16, critic_width=16, replay_capacity=64)
    learner = SimBaV2Hedger(config, **kwargs)
    previous = collect(learner, bank, 4, random_actions=True)
    learner.set_threshold(.01)
    before = copy.deepcopy(jax.device_get(learner.core._actor.params))
    info = learner.update(8)
    assert all(np.isfinite(value) for value in info.values())
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(before),
                                                      jax.tree_util.tree_leaves(learner.core._actor.params)))
    path = tmp_path / "checkpoint.pt"
    learner.save(path, runner={"previous": previous})
    clone = SimBaV2Hedger(config, **kwargs)
    saved = torch.load(path, weights_only=False)
    saved["learner"]["config"]["time_grid"].pop("trade_at_maturity")
    saved["learner"]["config"]["time_grid"].pop("step_days")
    saved["learner"]["config"].pop("settlement")
    clone.load_state_dict(saved["learner"])
    for deterministic in (True, False):
        a, tape_a = evaluate(learner, bank, seed=123, deterministic=deterministic, batch_size=4)
        b, tape_b = evaluate(clone, bank, seed=123, deterministic=deterministic, batch_size=4)
        torch.testing.assert_close(tape_a["terminal_loss"], tape_b["terminal_loss"], rtol=0, atol=0)
        assert a["cash_error"] < 2e-6 and b["cash_error"] < 2e-6
    previous_a = collect(learner, bank, 4, random_actions=False, previous=previous)
    previous_b = collect(clone, bank, 4, random_actions=False, previous=saved["runner"]["previous"])
    np.testing.assert_array_equal(previous_a["terminal_loss"], previous_b["terminal_loss"])
    learner.set_threshold(.02)
    clone.set_threshold(.02)
    learner.update(8)
    clone.update(8)
    for name in NETWORKS:
        for a, b in zip(jax.tree_util.tree_leaves(getattr(learner.core, name)),
                        jax.tree_util.tree_leaves(getattr(clone.core, name))):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(learner.core._rng, clone.core._rng)
    assert learner.agent.G_r_max >= abs(learner.replay.terminal_rewards(.02, .95)).max()
