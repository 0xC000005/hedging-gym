# Method adapters

These checkout modules provide small classical and trainable baselines around
the same market, observations, legal trades and cash ledger. The installable
`hedging_gym` package contains the environment and evaluator; learners remain
outside it.

See the [implementation appendix](../docs/baseline-methods.md) for the distinction
between author-code references, common-task transfers and diagnostic ports.

| Adapter | Behavior |
|---|---|
| `classical.py` | Current model-price sensitivities and bounded stock/option hedge targets |
| `controllers.py` | Common evaluation interface for classical controls and frozen policies |
| `policies.py` | Direct bounded holdings and learned no-transaction bands |
| `training.py` | Causal rollouts and a shared terminal expected-shortfall objective |
| `checkpoints.py` | Trusted-local snapshot files and RNG state; each learner owns its training state |
| `model_free.py` | Quantile D4PG and EX-D4PG with a generalized-Pareto tail |
| `sb3.py` | Stock SB3 algorithms through a thin batched VecEnv/action-coordinate adapter |
| `adaptation.py` | Full-network fine-tuning and task-embedding adaptation |
| `meta_pretraining.py` | First-order post-adaptation training, with a zero-inner-update ordinary control |
| `amago_adapter.py` | Optional official AMAGO/AMAGO-2 cross-book memory qualification |
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

### Acceleration

The common methods batch independent market paths, policy evaluations and
replay minibatches. DH, learned bands, hybrid policies and task embeddings can
keep their bank, ledger and gradients on GPU. Classical Greeks batch pricing
and differentiation; CEM batches roots, candidates and conditional scenarios.
For the separate adapter that imports the published MCTS/Trainer directly, see
[AlphaZero source-loop qualification](../docs/source-alphazero.md). It preserves
the donor's training loop and records its differences from the common ES task;
it is not yet a competitive baseline.

AlphaZero batches leaf pricing/inference across independent roots while each
tree keeps sequential selection and backup. Policy-only AlphaZero skips its
unused critic.

For SB3, `experiments.qualify_sb3 --device cpu|cuda` selects the PPO learner and
evaluator device. Its original NumPy VecEnv remains CPU-batched; the shared
evaluator uses the same SB3 action distribution directly on tensors, including
action clipping. Small MLPs can be faster on CPU, so time a representative
complete workload before choosing a device. Parallelize independent seeds
rather than oversubscribing every small matrix operation.

Acceleration must retain objectives, updates per transition and search budgets.
Check fixed-input actions, losses and gradients against the reference before a
long run. Record any changed collection ordering; faster execution does not by
itself establish a stronger or faithfully reproduced algorithm.

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

All methods use the same observed state, instrument set and accounting, and
report the same terminal expected shortfall. Source-preserving learners can
retain different training objectives; those are disclosed, not called identical
ES optimizers. The common benchmark is an **adaptation of the paper methods**,
not a claim to reproduce their original experiments.

The approved additions are **CrossQ**, **TQC** and **SimBaV2**, all with continuous
hedge actions. CrossQ and TQC use SB3-Contrib; SimBaV2 uses the authors' JAX
implementation. They are being qualified, not yet reported as trained or
competitive baselines. Their papers and source links are in
[related work](../docs/related-work.md), and their objective mappings are in the
[implementation appendix](../docs/baseline-methods.md). Rainbow is not part of
this addition because it would require a discretized hedge-action set.

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

For substantive RL training, use `experiments/qualify_model_free.py`, not the
eight-update integration recipe. It uses exploratory replay warmup and 32 replay
samples per newly collected transition. Actor, quantile critic and Pareto tail
have separate learning rates. Quantile-Huber smoothing is explicit in normalized
loss units; the repaired recipe adds action-derivative clipping and delayed actor
updates. These are disclosed common-task training choices, not a claim to match
every author setting. `hull_rl` refers to Cao/Hull's **2023 QR-D4PG** work, not
their original 2021 DDPG implementation. `dense_rewards=True` trains on changes in marked hedge wealth;
these telescope to the same terminal loss. Shifting the global risk threshold
by accumulated loss preserves the terminal ES objective. The intermediate
labels are detached: the actor still learns through its critic, not through
financial accounting gradients. This helps conditioning; it does not guarantee
a calibrated critic or a superior hedge.

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

The repaired adapter fits actor and value networks separately. Training-only
greedy rollouts calibrate the global ES threshold; fresh completed continuations
then refit value targets at that threshold before a checkpoint is saved. The
critic also receives marked hedge wealth computed from already observed cash,
holdings and marks. Counterfactual successor states broaden its action coverage.
These changes address stale targets and representation, not a proven performance
advantage. The value estimates describe the frozen greedy continuation; their
accuracy does not automatically carry over to a changed search controller.
Legacy checkpoints require their original implementation, not silent loading
into the revised architecture.

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

The runner's `--output-dir` saves initial policy weights, configuration, held-out
market bank and trade tapes. Adaptation results and final adapted policies are
separate files. Final ES is measured from actual held-out losses, not critics.

### Independent SB3 control

SB3 is optional: `uv sync --locked --group baselines`. The financial environment
does not depend on it. `SB3HedgingVecEnv` samples batches from a supplied training
bank and converts SB3's normalized actions into the same legal holding bounds.
It adds SB3's terminal-observation/autoreset convention, not another simulator.

```bash
uv run --frozen --group baselines python -m experiments.qualify_sb3 --run-dir /path/to/banks --output-dir /path/to/sb3-run
```

PPO uses complete-horizon rollouts with 512 episodes per update. At ES95 that is
about 26 expected tail outcomes, versus fewer than one for eight episodes. It
optimizes sampled-policy terminal risk; sampled and greedy deployment are
reported separately. Threshold calibration uses training paths only. Checkpoints
are standard SB3 ZIP files; reload checks verify identical greedy actions/losses,
not exact continuation of the custom environment's RNG state. SB3 does not
provide AlphaZero and is not a substitute for the distributional source methods.

### Checkpoint and resume training

Use `--checkpoint-dir` to retain full training state during a run, independently
of the final evaluation output. Each learner saves `latest.pt` after its first
update, every `--checkpoint-every` updates and at completion; `latest-early.pt`
preserves its first update. A snapshot includes the policy, relevant critics and
targets, optimizers, risk threshold, replay where used, RNG state and completed
work. The latest file is replaced atomically. These are trusted local PyTorch
files: do not load checkpoints from an untrusted source.

`--resume-from` resumes one selected method. Keep its original training bank,
configuration, seed and recipe; `--updates` specifies the new **total**, not an
additional number of steps. Only the total may be extended. Exact split/resume
checks cover same-device training; moving between CPU and GPU is not a promise
of bit-identical trajectories.

The Python trainers expose `checkpoint_path`, `checkpoint_every` and
`resume_from`. Chronological adaptation has a separate
`AdaptationUpdater.state_dict()` / `load_state_dict()` including a partially
completed update call. Save each stage separately so returning to A does not
overwrite its earlier evidence.

For repeated qualification on saved banks, see
`experiments/qualify_policies.py`, `qualify_adaptation.py`,
`qualify_model_free.py`, `qualify_alphazero.py` and `qualify_search.py`. These are experiment entry points,
not another training framework. Keep all banks, curves, checkpoints and trade
tapes outside the repository. A completed run is not automatically a qualified
competitive baseline; inspect actual hedges and development risk before a final
comparison.

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

For the separate official AMAGO donor qualification, see
[Context-aware hedging](../docs/amago-context.md). It keeps the common financial
environment, but currently trains a fixed-threshold RU objective; it is not yet
an ES-optimized competitive adaptation baseline.

For research comparisons, use development data to establish stable training,
then freeze budgets/settings and evaluate multiple seeds on a fresh, adequately
sized common bank. The current RL adapters still need convergence qualification;
working code alone is not a strong comparator. See [Benchmark](../docs/benchmark.md),
[Validation](../docs/validation.md) and [Related work](../docs/related-work.md).
