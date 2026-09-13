# Hedging Gym

![Research alpha](https://img.shields.io/badge/status-research_alpha-orange)
[![Original code: MIT](https://img.shields.io/badge/original_code-MIT-blue)](LICENSE)

A research library for option hedging, with a configurable financial environment,
classical and learned baselines, and paper benchmark configurations. All baselines
use the same market simulation, observations, legal trades, cash ledger and
terminal-loss evaluation.

| Area | What it provides | Guide |
|---|---|---|
| Environment | GBM, Heston and Bates markets; configurable instruments and execution; Gymnasium and PyTorch interfaces | [Getting started](docs/getting-started.md) |
| Baselines | Delta hedges, Deep Hedging, PPO, AlphaZero, adaptation and other named methods | [Baseline guide](docs/baselines.md) |
| Paper benchmarks | Bühler Heston and AlphaZero Heston/GBM configurations, runners and source comparisons | [Paper benchmarks](docs/paper-benchmarks.md) |

This is a **v0.1 research alpha**. The book contains one stock and
configurable claims at zero rates and dividends. Built-ins include European
options and a variance swap; users can supply their own instruments. Numerical and
API checks cover selected cases; Bates tail-risk precision remains incomplete.

## Start here

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
git clone https://github.com/0xC000005/hedging-gym.git
cd hedging-gym
uv sync --locked
uv run --frozen python -m hedging_gym.quickstart --device cpu --paths 64
```

The locked environment includes Python dependencies and QuantLib. The example
runs complete episodes with scripted trades; its small sample illustrates the
API rather than measuring hedging performance.

## Implemented methods

- Delta, delta-gamma and delta-variance, with optional fixed no-trade bands.
- Deep Hedging with direct pathwise training, learned no-transaction bands,
  full-network fine-tuning and task-embedding Adaptive Deep Hedging.
- Deep Bellman Hedging with actor-critic training and monetary-utility targets.
- Hull/Cao DDPG, QR-D4PG and EX-DRL ports; upstream
  PPO, CrossQ, TQC and SimBaV2 learners.
- AlphaZero source and common-environment
  implementations, CEM planning and hybrid policy optimization (HPO).
- GEPS conditioning, Belief-context DH and AMAGO,
  plus optional retrieval and pretraining extensions.

The [method catalogue](docs/baselines.md) links each module, paper, upstream source
and runner. It distinguishes published methods from local adaptations.
For Deep Bellman Hedging (`dbh`), see the
[objective mapping](docs/baseline-methods.md#training-objectives-are-not-interchangeable)
before comparing nested and terminal risk measures.

Run a small Deep Hedging / learned-band comparison, including classical controls:

```bash
uv run --frozen python -m benchmarks.baselines --methods dh ntb
```

There is no published cross-method leaderboard yet. The defaults exercise the
code; they are not paper-reproduction budgets. See
[reproducibility](docs/baseline-implementation.md) for seeds, saved checkpoints,
evaluation artifacts and comparison requirements. SB3-based methods need
`uv sync --locked --group baselines`; external author runtimes have separate
instructions in the catalogue.

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
docs/                usage, sources and reproducible validation
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

## Reference

- [Benchmark](docs/benchmark.md): defaults, financial equations, execution
  presets and observed regime changes.
- [Custom instruments](docs/custom-instruments.md): add a priced claim or choose
  a different settlement convention.
- [Validation](docs/validation.md): reproducible checks and their limits.
- [Related work](docs/related-work.md): papers and numerical implementations.
- [Contributing](CONTRIBUTING.md): add a method, validate it and submit a pull request.

## License

Original code and documentation use the [MIT License](LICENSE).
Incorporated third-party code retains its own license; see
[Third-party notices](THIRD_PARTY_NOTICES.md) for source credits and licenses.
