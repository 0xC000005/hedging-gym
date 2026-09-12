"""Replay actual donor Heston/GBM paths and actions through the common ledger.

This checks arithmetic with controlled inputs. It does not give search access
to future paths or assert equivalence of the two Heston simulation schemes.
The donor's game classes need scipy: uv run --with scipy ...
"""
import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from hedging_gym import config_from_dict, finance
from methods.alphazero import holding_grid


def compare(donor_path, model, seed):
    sys.path.insert(0, str(donor_path.resolve()))
    module, name = (("hedgerGame_TV_heston", "HedgerPlan_TV_heston") if model == "heston"
                    else ("hedgerGame_TV_pureGBMPaths", "HedgerPlan_TV_gbm"))
    native_class = getattr(importlib.import_module(f"hedger_TV.{module}"), name)
    config_path = Path(__file__).parent/"configs"/f"szehr-{model}-stock.json"
    config = config_from_dict(json.loads(config_path.read_text()))
    np.random.seed(seed)
    random.seed(seed)
    native = native_class({"reservoir": 1, "measureSampleEfficiency": True})
    state = native.getInitState()
    assert native.n_samples == config.n_steps
    assert native.dt == config.dt
    assert native.s0 == config.market.spot0
    assert native.strike == config.portfolio.liability.strike
    assert config.n_assets == 1 and native.transactionCosts == 0
    if model == "heston":
        for field in ("v0", "kappa", "theta", "sigma", "rho"):
            assert getattr(native, field) == getattr(config.market, field)
    else:
        assert native.sigma**2 == config.market.v0 and native.mu == config.market.mu

    grid = holding_grid(config, native.n_actions)[:, 0].numpy()
    source_grid = (np.arange(native.n_actions)-(native.n_actions-1)/2)*2/(native.n_actions-1)
    np.testing.assert_allclose(grid, source_grid, atol=5e-16, rtol=0)
    ledger = finance.initial_ledger(torch.ones(1, dtype=torch.float64), config)
    model_premium = float(ledger.cash[0])
    donor_premium = float(state[0, 2])
    # Harmonize this input only for the arithmetic comparison. Scientific runs
    # retain the common model price, not the source's hardcoded Heston premium.
    ledger = replace(ledger, cash=torch.tensor([donor_premium], dtype=torch.float64))
    native_cash, common_cash, native_holdings, common_holdings = [], [], [], []
    for date in range(config.n_steps):
        action = (7*date) % native.n_actions
        target = torch.tensor([[source_grid[action]]], dtype=torch.float64)
        marks = torch.tensor([[state[0, 3]]], dtype=torch.float64)
        ledger = finance.trade_step(ledger, target, marks, config)
        state, _ = native.getNextState(state, 1, action)
        native_cash.append(state[0, 2])
        common_cash.append(float(ledger.cash[0]))
        native_holdings.append(state[0, 1])
        common_holdings.append(float(ledger.positions[0, 0]))
    payoff = max(state[0, 3]-native.strike, 0.)
    loss = finance.liquidate(ledger, torch.tensor([[state[0, 3]]], dtype=torch.float64),
                            torch.tensor([payoff], dtype=torch.float64), config)["terminal_loss"]
    source_loss = float(payoff-native.stateValue(state))
    shared_loss = float(loss[0])
    # All values are ~1 in these normalized source tasks. This bound covers
    # double-precision accumulation; byte equality is measured separately.
    np.testing.assert_allclose(common_cash, native_cash, atol=2e-14, rtol=0)
    np.testing.assert_allclose(shared_loss, source_loss, atol=2e-14, rtol=0)
    def same_bytes(a, b):
        return np.asarray(a, dtype=np.float64).tobytes() == np.asarray(b, dtype=np.float64).tobytes()
    return dict(model=model, seed=seed, decisions=config.n_steps, days_per_year=config.time_grid.days_per_year,
        model_initial_premium=model_premium, donor_initial_premium=donor_premium,
        initial_premium_difference=model_premium-donor_premium,
        initial_cash_harmonized=True, action_grid_max_error=float(np.max(np.abs(grid-source_grid))),
        action_grid_byte_identical=same_bytes(grid, source_grid),
        cash_max_error=float(np.max(np.abs(np.array(common_cash)-native_cash))),
        cash_byte_identical=same_bytes(common_cash, native_cash),
        holdings_byte_identical=same_bytes(common_holdings, native_holdings),
        terminal_loss_max_error=abs(shared_loss-source_loss),
        terminal_loss_byte_identical=same_bytes(shared_loss, source_loss),
        raw_squared_loss=source_loss**2, common_objective_loss=float(config.risk.loss(loss)[0]),
        donor_clipped_reward=float(native.getGameEnded(state, 1)),
        path_source="donor; replayed only for the accounting audit",
        simulation_schemes=(f"donor Euler versus common {config.market.scheme}" if model == "heston"
                            else "both exact GBM steps"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for model in ("gbm", "heston"):
        print(f"CHECK: {model} donor paths, holdings and cash", flush=True)
        results.append(compare(args.donor, model, args.seed))
        print(json.dumps(results[-1]), flush=True)
    record = dict(source_commit=subprocess.check_output(
        ["git", "-C", str(args.donor), "rev-parse", "HEAD"], text=True).strip(),
        scope="fixed-path fixed-action accounting; not stochastic training equality", results=results)
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
