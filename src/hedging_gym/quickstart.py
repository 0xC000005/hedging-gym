"""Complete hedging episodes and a market-only A -> B -> A example.

Run ``python -m hedging_gym.quickstart --device cpu --paths 64`` after install.
The scripted trades demonstrate the API; there is no training or adaptation
performance claim, and the small sample ES is descriptive only.
"""
import argparse
from dataclasses import asdict
import time

import torch

from .benchmark import adaptation_configs, benchmark_config, operational_config
from .config import RiskConfig, TimeGrid
from .evaluation import empirical_es
from .gym_env import HedgingVectorEnv


def run(device="cpu", paths=64, *, model="heston", n_steps=30,
        days_per_year=252, risk_alpha=.95):
    if paths < 1:
        raise ValueError("paths must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    torch.set_num_threads(4)
    started = time.perf_counter()
    basic = benchmark_config(model=model,
        time_grid=TimeGrid(n_steps=n_steps, days_per_year=days_per_year),
        risk=RiskConfig(alpha=risk_alpha))
    operational = operational_config(basic, "operational_fixed", minimum_commission=.0001)
    operational = operational_config(operational, "operational_minimum_trade")
    operational = operational_config(operational, "operational_lots")
    stages = (("basic", basic), *[(f"operational/{name}", config)
                                 for name, config in adaptation_configs(operational)])
    print(f"Hedging Gym example; model={basic.market.model}; device={device}; CPU threads=4; "
          f"work=4 x {paths} complete {basic.n_steps}-date episodes; seeds=2026090600..2026090603",
          flush=True)
    print("Actions are target stock/call holdings, not trade sizes. Reward is zero "
          "until settlement, then cost-inclusive terminal P&L (negative loss). "
          f"ES at confidence {basic.risk.alpha:g} is the pooled upper tail of losses.", flush=True)
    print("Operational A -> B -> A changes only market variance; the execution overlay "
          "stays fixed. Scripted trades only: no learner or adaptation-performance test.", flush=True)
    for index, (name, config) in enumerate(stages):
        seed = 2026090600 + index
        print(f"START {name}: seed={seed}; config={asdict(config)}", flush=True)
        env = HedgingVectorEnv(paths, config, device=device)
        try:
            observed, _ = env.reset_tensor(seed=seed)
            target = torch.zeros((paths, config.n_assets), device=device)
            print(f"  observations={tuple(observed.shape)}; "
                  "legal targets=.1 / HOLD / .2 / HOLD, repeated", flush=True)
            for date in range(config.n_steps):
                if date % 2 == 0:
                    target = torch.full_like(target, .1 if date % 4 == 0 else .2)
                if not bool(env.action_mask_tensor(target[:, None, :]).all()):
                    raise RuntimeError("example target is infeasible")
                observed, reward, terminated, truncated, info = env.step_tensor(target)
                if (date + 1) % 10 == 0:
                    print(f"  {name}: {date + 1}/{config.n_steps} dates; "
                          f"total elapsed={time.perf_counter()-started:.2f}s", flush=True)
            if not bool(terminated.all()) or bool(truncated.any()):
                raise RuntimeError("example must finish a complete financial episode")
            print(f"DONE {name}: mean P&L={float(reward.mean()):.6f}; "
                  f"sample loss ES({config.risk.alpha:g})="
                  f"{empirical_es(info['terminal_loss'], config.risk.alpha):.6f}; "
                  f"mean costs={float(info['transaction_cost'].mean()):.6f} "
                  "(includes terminal liquidation)", flush=True)
        finally:
            env.close()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"DONE usage example: total elapsed={time.perf_counter()-started:.2f}s; "
          "small-sample outputs are not method-performance evidence", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--paths", type=int, default=64)
    parser.add_argument("--model", choices=("gbm", "heston", "bates"), default="heston")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--days-per-year", type=int, choices=(252, 365, 360), default=252)
    parser.add_argument("--risk-alpha", type=float, default=.95)
    arguments = parser.parse_args()
    run(arguments.device, arguments.paths, model=arguments.model, n_steps=arguments.steps,
        days_per_year=arguments.days_per_year, risk_alpha=arguments.risk_alpha)
