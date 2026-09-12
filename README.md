# Hedging Gym

![Research alpha](https://img.shields.io/badge/status-research_alpha-orange)

A research library for option hedging, with a configurable financial environment,
classical and learned baselines, and paper benchmark configurations. All baselines
use the same market simulation, observations, legal trades, cash ledger and
terminal-loss evaluation.

| Area | What it provides | Guide |
|---|---|---|
| Environment | GBM, Heston and Bates markets; configurable instruments and execution; Gymnasium and PyTorch interfaces | [Getting started](docs/getting-started.md) |
| Baselines | Delta hedges, Deep Hedging, PPO, AlphaZero, adaptation and other named methods | [Baseline guide](docs/baselines.md) |
| Paper benchmarks | Bühler Heston and AlphaZero Heston/GBM configurations, runners and source comparisons | [Paper benchmarks](docs/paper-benchmarks.md) |

This is a private **v0.1 research alpha**. The book contains one stock and
configurable claims at zero rates and dividends. Built-ins include European
options and a variance swap; users can supply their own instruments. Numerical and
API checks cover selected cases; Bates tail-risk precision remains incomplete.

## Start here

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed and
access to the repository:

```bash
git clone https://github.com/0xC000005/hedging-gym.git
cd hedging-gym
uv sync --locked
uv run --frozen python -m hedging_gym.quickstart --device cpu --paths 64
```

The locked environment includes Python dependencies and QuantLib. The example
runs complete episodes with scripted trades; its small sample illustrates the
API rather than measuring hedging performance.

## One complete batch

```python
import torch
from hedging_gym import HedgingVectorEnv, benchmark_config, empirical_es

config = benchmark_config(model="heston")
env = HedgingVectorEnv(64, config, device="cpu")
observed, _ = env.reset_tensor(seed=42)
targets = torch.zeros(
    (env.num_envs, config.n_assets), device=observed.device, dtype=observed.dtype
)
for _ in range(config.n_decisions):
    observed, reward, terminated, truncated, info = env.step_tensor(targets)

print("Terminal loss ES:", empirical_es(info["terminal_loss"], config.risk.alpha))
env.close()
```

Actions are target holdings. This example holds no hedges and shows the complete
settlement path. By default, reward is zero until settlement, then terminal P&L.
Optimizing mean reward and minimizing expected shortfall are different objectives.

## Repository layout

```text
src/hedging_gym/
  interfaces.py      public controller contract
  environment/       financial core, stepping, episodes and simulated branches
  adapters/          connectors to external learning libraries
  baselines/         one named entry point per complete method
    _shared/         reused learner mechanics, not additional methods
  extensions/        additions to existing policies and training procedures
  evaluation.py      common controller evaluation
benchmarks/          runnable comparisons and paper configuration files
tests/               financial, API and baseline checks
docs/                usage, sources and qualification evidence
```

These modules are installed together in the single `hedging-gym` package.
Methods use the environment directly or through library adapters;
the environment does not import learners. The package root keeps convenient
public imports such as `HedgingVectorEnv` and `benchmark_config`.

Each baseline has a named module, such as `delta.py`, `deep_hedging.py`,
`no_transaction_band.py` or `ppo.py`. Methods share the controller contract
and financial evaluator, while retaining their own learning procedures.
The [baseline guide](docs/baselines.md) links implementations, sources,
runners and qualification evidence, and explains [how to add a method](docs/baselines.md#add-a-method).

Run comparisons from a checkout with `uv run --frozen python -m benchmarks.baselines`.
Optional SB3 dependencies are available through the package's `baselines` extra;
in a checkout, use `uv sync --locked --group baselines`. Optional external donor
runtimes have their own setup instructions in the baseline guide.

## Reference

- [Benchmark](docs/benchmark.md): defaults, financial equations, execution
  presets and observed regime changes.
- [Custom instruments](docs/custom-instruments.md): add a priced claim or choose
  a different settlement convention.
- [Validation](docs/validation.md): reproducible checks and their limits.
- [Related work](docs/related-work.md): papers and numerical implementations.
