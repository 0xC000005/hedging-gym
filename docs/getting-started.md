# Getting started

## Install on Linux or WSL

Use a Linux terminal, including a Linux distribution under WSL. Install
[uv using its official instructions](https://docs.astral.sh/uv/getting-started/installation/),
then clone this private repository using an account with access:

```bash
git clone https://github.com/0xC000005/hedging-gym.git
cd hedging-gym
uv sync --locked
```

`uv sync --locked` creates the project environment, installs the package and
development dependencies, and checks that the lockfile matches the project.
QuantLib is a project dependency and is installed with the other packages;
there is no separate QuantLib setup step. Python 3.12–3.13 is supported by the
package metadata; Linux with Python 3.12 is the qualification platform. See
[uv's project workflow](https://docs.astral.sh/uv/guides/projects/) for environment
and lockfile behavior.

After this sync, `uv run --frozen` uses the existing lockfile without updating
it. It does not recheck whether dependency declarations have changed; run
`uv sync --locked` again after changing project dependencies.

Run the example and checks as separate commands so each result is visible:

```bash
uv run --frozen python -m hedging_gym.quickstart --device cpu --paths 64
uv run --frozen python -m hedging_gym.validate --device cpu
uv run --frozen pytest
```

The example runs scripted trades in the basic market and an observed A → B → A
sequence. The validator compares prices with QuantLib, reconstructs complete
cash ledgers, and checks the scalar Gymnasium interface. The test suite adds
financial, configuration, execution and method checks. Their scope is explained
in [Validation](validation.md).

Start with CPU. If a compatible CUDA device and PyTorch installation are
available, select `--device cuda` for the example or validator. Device, batch
size and integration resolution are part of reproducibility; equal seeds across
devices do not promise identical paths.

On Linux, the locked PyTorch package includes CUDA libraries even for CPU runs.
A first installation can download several gigabytes; `--device` selects where
computation runs, not which dependencies are installed.

## Configure an experiment

`benchmark_config()` supplies one stock, one ATM hedge call and an ATM call
liability. A configuration is composed from independent parts:

| Part | Responsibility |
|---|---|
| `GBMConfig`, `HestonConfig`, `BatesConfig` | Market dynamics and initial market state |
| `TimeGrid` | Number of decisions and annualization convention |
| `PortfolioConfig` and `EuropeanOption` | Liability, its quantity and hedge options |
| `ExecutionConfig` | Position limits, fees, minimum trades and lots |
| `RiskConfig` | Expected-shortfall confidence level |
| `HedgingConfig` | The complete, validated experiment configuration |

For example, use GBM, a 20-day grid with 365-day annualization, a put liability,
stock-only hedging and a 99% risk level:

```python
from hedging_gym import (
    GBMConfig, TimeGrid, EuropeanOption, PortfolioConfig,
    ExecutionConfig, RiskConfig, benchmark_config,
)

grid = TimeGrid(n_steps=20, days_per_year=365)
portfolio = PortfolioConfig(
    liability=EuropeanOption(strike=1.0, maturity=grid.horizon, kind="put"),
    hedges=(),
    liability_quantity=1.0,
)
config = benchmark_config(
    model=GBMConfig(v0=0.04),
    time_grid=grid,
    portfolio=portfolio,
    execution=ExecutionConfig(
        holding_lower=-1.0, holding_upper=2.0, proportional=0.0005,
    ),
    risk=RiskConfig(alpha=0.99),
)
```

Option maturities are in years. `days_per_year` supports 252, 365 and 360;
these are regular model grids, not holiday calendars. The episode ends at the
liability maturity. Hedge options may expire at that endpoint or later;
intermediate expiry is not supported. Rates and dividends currently must be zero.
Contract maturities must lie on whole days of the selected year clock.

Execution fields accept a scalar for all tradable instruments or a tuple in
stock-then-hedge order. Scalars make stock-only and larger hedge books easy to
construct; tuples express instrument-specific rules. Always derive action and
observation sizes from the configured environment rather than copying dimensions
from an example. See [Benchmark](benchmark.md) for the complete financial contract.

## Run and evaluate a controller

The [README example](../README.md#one-complete-batch) uses the tensor interface.
`HedgingEnv` supplies a scalar Gymnasium environment. `HedgingVectorEnv` supplies
batched NumPy `reset`/`step` and tensor `reset_tensor`/`step_tensor` interfaces.
They use the same market and ledger. Batch members finish together; reset
explicitly before another episode.

An action specifies the desired holdings, not the trade increment. Holdings
must satisfy both their outer bounds and the rules for trading from the current
portfolio. Use `action_mask` or `action_mask_tensor` to inspect candidate targets.
Invalid trades raise errors. Fixed tickets and lot constraints are discrete;
continuous gradients alone do not optimize their activation decisions.

For evaluation without a Gym loop, pass a controller and a generated bank:

```python
import torch
from hedging_gym import benchmark_config, evaluate_controller
from hedging_gym.finance import generate_market_bank

config = benchmark_config(model="gbm")
bank = generate_market_bank(config, n_paths=64, seed=42)

def fixed_stock(observed, ledger, time_index, config):
    targets = ledger.positions.clone()
    targets[:, 0] = 0.1
    return targets

metrics, tape = evaluate_controller(fixed_stock, bank)
print(metrics)
```

The controller receives current observations, ledger state, date index and
configuration. Treat these inputs as read-only: do not modify observations or
ledger tensors in place. Clone holdings when adjusting them, as above; only
the environment updates the actual ledger. Keep learner weights frozen during
evaluation. The returned tape records
actual holdings, terminal losses and execution costs for cash reconstruction.
For a learned comparison, use separate training and evaluation paths and pool
complete losses when calculating ES; minibatch ES values cannot be averaged.

The trainable baseline example is a separate checkout command:

```bash
uv run --frozen python -m experiments.baselines
```

Read the [method guide](../methods/README.md) before interpreting its results or
selecting execution rules for a continuous policy.
