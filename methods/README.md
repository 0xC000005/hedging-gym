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
| `model_free.py` | Quantile D4PG and EX-D4PG with a generalized-Pareto tail |
| `adaptation.py` | Full-network fine-tuning and task-embedding adaptation |
| `alphazero.py` | Stochastic PUCT, learned policy/value and search-improvement training |
| `hybrid.py` | Discrete HOLD/TRADE choices with pathwise sizing and categorical PPO |
| `planning.py` | CEM root-action improvement, feedback rollouts and optional gradient refinement |

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

## Full comparison set

All methods use the same observed state, instrument set, accounting and terminal
expected shortfall. The common benchmark is an **adaptation of the paper
methods**, not a claim to reproduce their original experiments.

| Command name | Learning/control mechanism | Source and main transfer change |
|---|---|---|
| `dh` | Differentiate terminal loss through trading decisions | [Bühler et al.](https://arxiv.org/abs/1802.03042); shared multi-instrument observations and ES objective |
| `ntb` | Learn acceptable holding bands; trade only to a boundary | [Imaki et al.](https://arxiv.org/abs/2103.01775); learned centers for each configured asset |
| `hull_rl` | Model-free quantile D4PG: replay, target networks and critic-based actor gradients | [Rotman/Hull code](https://github.com/rotmanfinhub/gamma-vega-rl-hedging/tree/77dc48326da000d983b1fb750edb2177e38c75fd); terminal ES replaces the original conditional risk objective/option-arrival task |
| `exdrl` | Quantile body plus fitted generalized-Pareto loss tail | [EX-DRL code](https://github.com/pmalekzadeh/EX-DRL/tree/f1abe99df7fa9efaa65af6b9dd416c3425c64098); same objective transfer, batched PyTorch rather than Acme/Reverb |
| `finetune_dh` | Continue updating all DH weights on current-market paths | Ordinary DH before adaptation; weights and adaptation optimizer persist through market changes |
| `adaptive_dh` | Pretrain shared weights/task embeddings, then fit only a new embedding | [Schmid–Oeltz, §2.2](https://arxiv.org/html/2504.16436v1#S2.SS2); ES and multiple instruments replace squared error and the frictionless stock-only book |
| `alphazero` | Search → visit-policy/terminal-value fitting → improved search | [Szehr's author code](https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba); stochastic chance nodes, configured holding grid and global terminal ES |
| `cem` | Fit a sampling distribution to promising current hedge targets | [Cross-entropy optimization](https://doi.org/10.1023/A:1010091220143); fresh conditional scenarios and frozen DH feedback to expiry |
| `hpo` | Learn trade/no-trade exploration and differentiate conditional trade sizes | [HPO author code](https://github.com/MatiasAlvo/hybrid-rl/tree/e48ae86da1e8f14c93cbb56e48d87f8674228659); live-history mixed gradient, then categorical PPO; complete-return ES instead of the native LQR/inventory training recipe |
| `hpo_cem` | Use the same CEM planner with hybrid-policy proposals/continuation | Our learned-guidance comparison, not a separate published method |
| `hpo_gradient` | Add action-gradient refinement to `hpo_cem` | Our search ablation; refinement work is counted separately |

The model-free learners do **not** backpropagate through accounting. Their
critics estimate future terminal loss distributions; EX-DRL's Pareto component
participates in both targets and actor optimization. A single global threshold
keeps their objective aligned with the other methods rather than silently
changing to a different conditional-tail objective at each date.

HPO retains the gradient through earlier sizing into later categorical-choice
probabilities. Extra PPO passes detach those histories; no straight-through
trade/no-trade gradient is substituted. It uses complete Monte Carlo returns,
four configurable PPO epochs and a finite episode, not the source's full LQR
recipe. Sampling during training and greedy evaluation are different policies;
the runner explicitly evaluates the latter.

AlphaZero expands actual decision/chance trees and learns from visit counts and
realized episode returns. The default holding grid is coarse; `--grid-points`
controls it, or pass an explicit action table to the Python API. Refining it and
allowing sufficient search/training are necessary before a competitive claim.
Market outcomes are averaged, never treated as actions to optimize. Inference
and conditional pricing are batched; traversal is on CPU. `simulations=0` in
`alphazero_controller` evaluates its learned policy without search.

CEM improves the **current** action and uses a frozen feedback policy afterward.
It is not a full open-loop MPPI controller. `cem` and the HPO variants disclose
their different continuations; compare unguided/guided `RolloutPlanner` instances
with the **same** continuation to isolate proposal guidance alone. All variants
score candidates on shared conditional paths independent of evaluation paths.

### Exercise every adapter

```bash
uv run --frozen python -m experiments.baselines --methods all --device cuda --steps 30 --train-paths 256 --eval-paths 64 --updates 4 --batch-size 16 --search-batch-size 8 --search-simulations 8 --search-candidates 8 --search-scenarios 8 --threads 2 --adaptation-updates 4 --output-dir /tmp/hedging-baselines
```

Use `--device cpu` without CUDA. This command is an integration run: four
training batches and 64 evaluation paths do not rank the algorithms. In
particular, one `update` means a DH optimizer step, an RL collection plus replay
updates, or an AlphaZero self-play batch plus training. Equal update counts do
**not** mean equal compute. Saved metadata records their actual work and time.

The runner saves initial policy weights, configuration, held-out market bank and
trade tapes. Adaptation results and final adapted policies are separate files;
these are evaluation checkpoints, not exact optimizer/RNG resumption snapshots.
Keep output directories outside the repository. Training thresholds are retained,
but final ES is always measured from actual held-out losses, not critic outputs.

### Operational and adaptation comparisons

`--preset operational_fixed` or `--preset operational_minimum_fee` adds execution
frictions without changing the market or book. HPO and continuous RL/DH do not
yet parameterize lots or minimum order sizes; the AlphaZero Python adapter can
mask legal grid actions. Do not silently round continuous baselines into those
experiments.

`--adaptation-updates N` runs market-only A → B → A for `finetune_dh` and
`adaptive_dh`, with N optimizer steps per stage. Execution settings stay fixed.
Both receive the same separately seeded stage training and evaluation banks.
The multitask source set is disclosed: baseline A and a market with 0.8 times
its initial/long-run variance, not future evaluation B. Pre-update return-A
performance measures forgetting; post-update performance measures recovery.
No policy is restored using a hidden regime lookup.

For research comparisons, use development data to establish stable training,
then freeze budgets/settings and evaluate multiple seeds on a fresh, adequately
sized common bank. The current RL adapters still need convergence qualification;
working code alone is not a strong comparator. See [Benchmark](../docs/benchmark.md),
[Validation](../docs/validation.md) and [Related work](../docs/related-work.md).
