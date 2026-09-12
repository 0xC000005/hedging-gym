# Hedging Gym

![Research alpha](https://img.shields.io/badge/status-research_alpha-orange)

A batched environment for option-hedging research, with GBM, Heston and Bates
markets, Gymnasium and PyTorch interfaces, one cash ledger, and terminal-loss
evaluation. Market parameters, portfolio, time grid, execution rules and risk
level are configured separately.

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

- [Getting started](docs/getting-started.md): Linux/WSL setup, configuration,
  tensor and Gym interfaces, and first checks.
- [Benchmark](docs/benchmark.md): defaults, financial equations, execution
  presets and observed regime changes.
- [Paper configurations](docs/paper-benchmarks.md): Bühler Heston, AlphaZero
  Heston and AlphaZero GBM.
- [Custom instruments](docs/custom-instruments.md): add a priced claim or choose
  a different settlement convention.
- [Validation](docs/validation.md): reproducible checks and their limits.
- [Methods](methods/README.md): classical controls and trainable baseline adapters.
- [Integration review](docs/integration-review.md): retained research branches,
  historical checkpoint compatibility and current qualification boundaries.
- [Related work](docs/related-work.md): papers and numerical implementations.

The installable package contains the environment and evaluator. Learners in
`methods/` and runnable comparisons in `experiments/` are used from a checkout.
