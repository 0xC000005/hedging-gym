# Method adapters

These checkout modules provide small classical and trainable baselines around
the same market, observations, legal trades and cash ledger. The installable
`hedging_gym` package contains the environment and evaluator; learners remain
outside it.

| Adapter | Behavior |
|---|---|
| `classical.py` | Current model-price sensitivities and bounded stock/option hedge targets |
| `controllers.py` | Common evaluation interface for classical controls and frozen policies |
| `policies.py` | Direct bounded holdings and learned no-transaction bands |
| `training.py` | Causal rollouts and a shared terminal expected-shortfall objective |

The direct policy follows the
[Deep Hedging](https://arxiv.org/abs/1802.03042v1) approach. The band policy adapts
[Imaki et al.](https://arxiv.org/abs/2103.01775v1) and
[PFHedge's pinned example](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/README.md):
it learns a center and bounded widths per instrument and keeps existing holdings
within the band. These are adaptations of established methods. Classical
sensitivities match local price changes rather than minimizing terminal ES;
variance sensitivity means `d/dv`, not `d/dsqrt(v)`.

## Run the example

From the repository root after `uv sync --locked`:

```bash
uv run --frozen python -m experiments.baselines
uv run --frozen python -m experiments.baselines --help
```

The default CPU run uses the basic benchmark, 128 training paths and a separate
128-path evaluation bank. It trains each selected learned policy for eight
updates with minibatches of 32 and hidden layers of size 32, 32. The standard
comparison includes delta, a fixed stock-quantity delta band, delta-gamma,
delta-variance, direct Deep Hedging and a learned band. Policy seed is 7;
training and evaluation bank seeds are 1101 and 2201.

Use `--steps`, `--days-per-year` and `--risk-alpha` to change the decision count,
annualization convention and terminal-risk confidence independently. For example:

```bash
uv run --frozen python -m experiments.baselines --model gbm --steps 20 --days-per-year 365 --risk-alpha 0.99
```

`--device cuda` selects a compatible CUDA installation. `--output-dir` saves
configuration, weights, the held-out market bank and trade tapes to a directory
you choose outside Git. Policies are frozen before generating the evaluation
bank. Progress reports configuration, seeds, device, workload and timings.

This small run demonstrates training and evaluation. It has too little training
and too few tail observations to support a hedging-quality claim. Interpret
timings separately for bank preparation, training and evaluation; equal seeds
across devices do not guarantee equal paths.

## Execution and evaluation limits

Derive policy input and output sizes from the configured observation schema and
asset count. Evaluate all methods with the same portfolio, execution rules,
capital, risk level and fresh market paths.

The policy constructors take the configuration directly. `from_env` uses the
same configuration as an existing scalar or batched environment:

```python
from hedging_gym import HedgingEnv, benchmark_config
from methods.policies import DirectDHPolicy

env = HedgingEnv(benchmark_config(model="gbm"))
policy = DirectDHPolicy.from_env(env, hidden=(32, 32))
env.close()
```

Saved policy state retains its observation and instrument schema. Reuse across
observed parameter changes is allowed; a different schema must be matched by a
corresponding policy, even when the total feature count happens to be equal.

The continuous learned policies do not parameterize discrete lot sizes or
minimum orders; training rejects those settings. Hard fixed-ticket activation
also has no ordinary pathwise gradient. A method needs an appropriate treatment
of those decisions before results under discontinuous costs can be interpreted
as optimized hedging. Targets enter the common ledger without implicit rounding.

The observed A → B → A evaluator can retain a caller's learner and update
callback across episodes, but this example is not a trained adaptation study.
See [Benchmark](../docs/benchmark.md), [Validation](../docs/validation.md) and
[Related work](../docs/related-work.md) for the shared contract and evidence scope.
