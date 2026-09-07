"""Independent price, accounting and environment checks.

Run ``python -m hedging_gym.validate --device cpu`` after install.
Checks selected analytic prices, a complete independently reconstructed ledger,
and Gymnasium's scalar API. Bates remains experimental; these few price states
do not qualify its transition law or a wider pricing domain.
"""
import argparse
import time

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from . import finance
from .benchmark import benchmark_config, operational_config
from .evaluation import evaluate_controller
from .gym_env import HedgingEnv


def scripted_targets(observed, ledger, time_index, config):
    """Legal .1/HOLD/.2/HOLD in each instrument; final holdings are nonzero."""
    if time_index % 2:
        return ledger.positions
    return torch.full_like(ledger.positions, .1 if time_index % 4 == 0 else .2)


def run(device="cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    torch.set_num_threads(4)
    started = time.perf_counter()
    print(f"FINANCIAL VALIDATION; device={device}; CPU threads=4; "
          "9 float64 prices, 8 complete 30-date Heston tapes, scalar Gym checker; "
          "market seed=2026090610; evaluation seed=2026090611", flush=True)
    # Fixed ordinary, short-expiry and higher-variance states; exact n/252 grid.
    states = ((1., .04, 30, 1.), (1., .04, 1, 1.), (1., .09, 60, 1.))
    for model in ("heston", "gbm", "bates"):
        config = benchmark_config(model=model)
        prices = finance.call_price(
            torch.tensor([s for s, v, n, k in states], dtype=torch.float64, device=device),
            [v for s, v, n, k in states], [n/252 for s, v, n, k in states],
            [k for s, v, n, k in states], config.market).cpu().numpy()
        reference = np.array([finance.quantlib_option_price(s, v, n/252, k, config.market)
                              for s, v, n, k in states])
        np.testing.assert_allclose(prices, reference, atol=1e-8, rtol=1e-7)
        print(f"PASS {model} prices: 3 states; max error={np.max(np.abs(prices-reference)):.3g}; "
              "atol=1e-8, rtol=1e-7" + ("; Bates experimental" if model == "bates" else ""),
              flush=True)
    config = operational_config(benchmark_config(), "operational_fixed",
        minimum_commission=(.0001, .0001), minimum_trade=(.01, .1), trade_lot=(.001, .1))
    print("START ledger: tickets=.0001/.0001; minimum fees=.0001/.0001; "
          "minimum trades=.01/.1; lots=.001/.1; targets=.1/HOLD/.2/HOLD", flush=True)
    bank = finance.generate_market_bank(config, 8, 2026090610)
    _, tape = evaluate_controller(scripted_targets, bank, device=device, batch_size=8,
        mode_seed=2026090611, label="Operational Heston accounting", progress=True)
    positions = tape["positions"].numpy().astype(np.float64)
    trades = np.diff(positions, axis=1, prepend=np.zeros_like(positions[:, :1]),
                     append=np.zeros_like(positions[:, :1]))
    if not (np.isfinite(positions).all() and (positions >= config.execution.vector("holding_lower", config.n_assets)).all()
            and (positions <= config.execution.vector("holding_upper", config.n_assets)).all()
            and ((trades[:, :-1] == 0) |
                 (np.abs(trades[:, :-1]) >= config.execution.vector("minimum_trade", config.n_assets))).all()):
        raise AssertionError("independent tape check: invalid holdings or minimum trade")
    # This fixed .1/.2 probe allows only float32 representation error in lots.
    lots = np.asarray(config.execution.vector("trade_lot", config.n_assets))
    np.testing.assert_allclose(trades, np.round(trades/lots)*lots, atol=2e-7, rtol=0)
    reference = finance.numpy_ledger(bank.marks.numpy(), positions,
        bank.liability[:, 0].numpy(), bank.liability[:, -1].numpy(), config)
    for key in ("terminal_loss", "transaction_cost", "turnover", "tickets"):
        np.testing.assert_allclose(tape[key].numpy(), reference[key], atol=2e-6, rtol=0)
    if bool((tape["positions"][:, -1] == 0).any()):
        raise AssertionError("the accounting probe must exercise terminal liquidation")
    print("PASS independent NumPy tape legality and ledger: loss, costs, turnover, tickets; "
          "includes terminal liquidation; absolute tolerance=2e-6", flush=True)
    if device == "cuda":
        _, cpu_tape = evaluate_controller(scripted_targets, bank, device="cpu", batch_size=8,
            mode_seed=2026090611, label="CPU replay", progress=True)
        torch.testing.assert_close(tape["terminal_loss"], cpu_tape["terminal_loss"], atol=2e-6, rtol=0)
        print("PASS CPU/CUDA terminal losses on the same bank; absolute tolerance=2e-6", flush=True)
    env = HedgingEnv(benchmark_config())
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    print("PASS scalar Gymnasium API (CPU, rendering skipped)", flush=True)
    print(f"DONE: total elapsed={time.perf_counter()-started:.2f}s; implementation checks only, "
          "not a transition-distribution or performance certificate; Bates experimental", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run(parser.parse_args().device)
