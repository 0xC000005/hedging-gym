# Paper configurations

Three examples use the same environment, instrument interface and cash ledger.
The market, hedge instruments, trading dates, fees and objective are independent
settings—not separate environment implementations.

| Configuration | Market and calendar | Hedge instruments | Objective |
|---|---|---|---|
| [Bühler Heston](../experiments/configs/buehler-heston.json) | Heston QE-M, spot 100, 30 decisions, 365-day clock | Stock and variance swap, with cash | ES50; MSE is an available variation |
| [AlphaZero Heston](../experiments/configs/szehr-heston-stock.json) | Heston QE-M, spot 1, 60 decisions, 365-day clock | Stock and cash | MSE |
| [AlphaZero GBM](../experiments/configs/szehr-gbm-stock.json) | GBM, spot 1, volatility 30%, 30 decisions, 365-day clock | Stock and cash | MSE |

All three have zero trading fees. The AlphaZero configurations use the released
**Szehr/minimalHedger** environments, not the later Maggiolo non-convex-cost paper.
The supplied JSON files can be edited or loaded directly:

```python
import json
from pathlib import Path
from hedging_gym import HedgingVectorEnv, config_from_dict

path = Path("experiments/configs/buehler-heston.json")
config = config_from_dict(json.loads(path.read_text()))
env = HedgingVectorEnv(128, config, device="cpu")  # "cuda" for a GPU batch
observed, _ = env.reset_tensor(seed=42)
```

The equivalent Python constructors are `buehler_heston()`,
`szehr_market("heston")` and `szehr_market("gbm")` in
`hedging_gym.paper_benchmarks`. For example:

```python
from dataclasses import replace
from hedging_gym import RiskConfig, SettlementConfig
from hedging_gym.paper_benchmarks import buehler_heston

config = buehler_heston(objective="mse")
config = replace(config, settlement=SettlementConfig(mode="mark_to_market"))
```

Changing these choices creates an explicitly configured variation; it does not
retroactively change what a published experiment tested.

## Bühler: what matches, what differs

[Deep Hedging, §§5.1–5.4](https://arxiv.org/html/1802.03042#S5) specifies a Heston
market with `v0=theta=.04`, `kappa=1`, `sigma=2`, `rho=-.7`, zero rates and
dividends, and one short ATM European call. The second hedge is the positive-price,
unannualized variance leg, paying accumulated variance at maturity. Its value is
past accumulated variance plus the conditional expected remaining variance.
It is not a variance option: there is no option-style strike threshold on that
variance payoff. The class is named `VarianceSwap` to match the paper's terminology.

The preset has no artificial holding cap: `holding_lower` and `holding_upper`
are `null`. In particular, limiting raw variance-swap quantities to `[-1,1]`
would change the experiment substantially. It uses the model option premium
as initial cash, not the rounded number printed in the paper. Terminal fees
are waived. The classical delta–variance control supports this instrument.

Our sampler uses batched QE-M with a trapezoidal variance integral. QE-M is our
numerical choice; the paper uses a different Heston sampling scheme. Its
martingale correction does not make the joint-path simulation or variance
integral exact. Refining simulation substeps preserves trading dates and checks
numerical convergence. Plain QE remains available through `scheme="qe"`.

## AlphaZero: what matches, what differs

The [released Heston](https://github.com/plan64/minimalHedger_AlphaZero/blob/3111c378fcd17e45f94d2fc668a3aa117126ecba/hedger_TV/hedgerGame_TV_heston.py)
and [GBM](https://github.com/plan64/minimalHedger_AlphaZero/blob/3111c378fcd17e45f94d2fc668a3aa117126ecba/hedger_TV/hedgerGame_TV_pureGBMPaths.py)
tasks supply the market, book, calendar and stock bounds `[-1,1]`. Source search
uses a 21-point target-holdings grid. The Gym's continuous action space allows
other methods to select holdings; use the same grid for a discrete comparison.

Our Heston uses QE-M instead of the donor's Euler discretization. We calculate
the initial premium rather than hardcode `0.021`. MSE is the unclipped squared
terminal loss; the source has a scaled, clipped reward. The
[source adapter and cash-replay check](source-alphazero.md) document these
differences. Matching financial tasks does not claim matching trained results.

## The separate 2025 non-convex GBM variant

`maggiolo_gbm(step_days=...)` composes the task from
[Maggiolo et al., §4.2.2 and Appendix D.2.2](https://arxiv.org/html/2510.01874v2):
spot/strike 5, volatility .25, physical drift .03125, twenty holdings
`0,.05,…,.95`, and a capped per-share fee `min(.25*abs(trade), .05)`.
It starts with cash .4 and stock .4. Four market moves are followed by settlement;
five actions include a trade at maturity. Remaining inventory is marked to
market without another liquidation fee.

The paper does **not** specify the physical time increment. The constructor
requires it explicitly: `step_days=365` means a unit-year reconstruction on the
365-day clock, not a verified paper setting. This variant is separate from the
three source configurations above.

## What is ready to use

The scalar, vector and tensor interfaces share these instruments and settlement
rules. Tests cover the variance-swap equation and Greeks, independent cash
reconstruction, maturity-date trading, config round trips and a user-defined
instrument. Run:

```bash
uv run --frozen --group baselines pytest -q tests/test_paper_benchmarks.py tests/test_source_alphazero.py
```

Algorithm compatibility is separate. Direct Deep Hedging accepts unbounded
positions; the current band/HPO parameterizations need finite bounds. Discrete
search also needs a finite grid. Choosing such bounds for the Bühler task is
an additional experiment setting, not part of its original unconstrained task.
