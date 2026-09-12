# Baselines

Complete hedgers have named modules in `hedging_gym.baselines`. Reusable training
and adaptation additions live in `hedging_gym.extensions`. Both use the common
financial environment and evaluator. Module headers link to the paper, pinned
upstream code where applicable, and implementation notes.

## Method catalogue

The implementation column distinguishes imported learners, paper-based ports
and local adaptations. These are source descriptions, not separate method
families or performance rankings. Detailed source changes are in the
[implementation appendix](baseline-methods.md); dated results remain in
[qualification evidence](baseline-implementation.md).

| Method / implementation | Source and implementation | Runner | Evidence |
|---|---|---|---|
| [Delta](../src/hedging_gym/baselines/delta.py) | [model sensitivities](benchmark.md). Local sensitivity control; optional fixed band. | [baselines](../benchmarks/baselines.py), included classical control; `--delta-band` selects a fixed band | [Financial checks](validation.md) |
| [Delta-gamma](../src/hedging_gym/baselines/delta_gamma.py) | [model sensitivities](benchmark.md). Local two-sensitivity control. | [baselines](../benchmarks/baselines.py), included classical control | [Financial checks](validation.md) |
| [Delta-variance](../src/hedging_gym/baselines/delta_variance.py) | [model sensitivities](benchmark.md). Local two-sensitivity control. | [baselines](../benchmarks/baselines.py); [variance-swap task](paper-benchmarks.md) | [Financial checks](validation.md) |
| [Deep Hedging (`dh`)](../src/hedging_gym/baselines/deep_hedging.py) | [Bühler et al.](https://arxiv.org/abs/1802.03042v1). Paper-mechanism PyTorch implementation. | [baselines](../benchmarks/baselines.py), `--methods dh`; [qualify_policies](../benchmarks/qualify_policies.py) | [Qualification](baseline-implementation.md) |
| [No-transaction bands (`ntb`)](../src/hedging_gym/baselines/no_transaction_band.py) | [Imaki et al.](https://arxiv.org/abs/2103.01775v1), [PFHedge example](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/README.md). Paper-mechanism implementation using the referenced band construction. | [baselines](../benchmarks/baselines.py), `--methods ntb`; [qualify_policies](../benchmarks/qualify_policies.py) | [Qualification](baseline-implementation.md) |
| [Full-network fine-tuning (`finetune_dh`)](../src/hedging_gym/baselines/finetune_dh.py) | ordinary DH control. Ordinary full-network updating control. | [qualify_adaptation](../benchmarks/qualify_adaptation.py); [matched update comparison](fast-adaptation.md#does-freezing-the-shared-network-help) | [Qualification](baseline-implementation.md) |
| [Adaptive Deep Hedging (`adaptive_dh`)](../src/hedging_gym/baselines/adaptive_deep_hedging.py) | [Schmid–Oeltz](https://arxiv.org/abs/2504.16436v1). Paper-mechanism task-embedding implementation. | [qualify_adaptation](../benchmarks/qualify_adaptation.py); [fast adaptation](fast-adaptation.md) | [Qualification](baseline-implementation.md) |
| [Hull/Cao 2021 DDPG](../src/hedging_gym/baselines/hull_ddpg.py) | [author code](https://github.com/rotmanfinhub/deep-hedging-research/tree/b4d031a185fe2547dd81ad7a67081f6dbe52c5bc). Source-backed PyTorch port with disclosed corrections. | [qualify_hull_ddpg](../benchmarks/qualify_hull_ddpg.py), saved training/development banks | [Original-objective evidence](baseline-implementation.md) |
| [PPO](../src/hedging_gym/baselines/ppo.py) | [Schulman et al.](https://arxiv.org/abs/1707.06347). Upstream Stable-Baselines3 learner. | [qualify_sb3](../benchmarks/qualify_sb3.py), saved banks | [Qualification](baseline-implementation.md) |
| [CrossQ](../src/hedging_gym/baselines/crossq.py) | [Bhatt et al.](https://arxiv.org/abs/1902.05605v4). Upstream SB3-Contrib learner. | [qualify_off_policy](../benchmarks/qualify_off_policy.py), `--algorithm crossq` | [Completed, weak initial runs](baseline-implementation.md) |
| [TQC](../src/hedging_gym/baselines/tqc.py) | [Kuznetsov et al.](https://proceedings.mlr.press/v119/kuznetsov20a.html). Upstream SB3-Contrib learner. | [qualify_off_policy](../benchmarks/qualify_off_policy.py), `--algorithm tqc` | [Completed, weak initial runs](baseline-implementation.md) |
| [SimBaV2](../src/hedging_gym/baselines/simbav2.py) | [Lee et al.](https://arxiv.org/abs/2502.15280v2), [official learner](https://github.com/DAVIAN-Robotics/SimbaV2). Imported official JAX learner. | [qualify_simbav2](../benchmarks/qualify_simbav2.py), external donor and saved banks | [Completed, weak initial runs](baseline-implementation.md) |
| [AlphaZero source loop](../src/hedging_gym/baselines/source_alphazero.py) | [pinned Szehr code](https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba). Imported source search/training; policy-argmax deployment. | [qualify_source_alphazero](../benchmarks/qualify_source_alphazero.py); [Heston/GBM configurations](paper-benchmarks.md) | [Source-loop checks and limits](source-alphazero.md) |
| [Hybrid policy optimization (`hpo`)](../src/hedging_gym/baselines/hpo.py) | [Alvo et al. code](https://github.com/MatiasAlvo/hybrid-rl/tree/e48ae86da1e8f14c93cbb56e48d87f8674228659). Source-backed mixed-gradient finance adaptation. | [baselines](../benchmarks/baselines.py), `--methods hpo`; [qualify_policies](../benchmarks/qualify_policies.py) | [Qualification](baseline-implementation.md) |
| [CEM (`cem`)](../src/hedging_gym/baselines/cem.py) | [cross-entropy optimization](https://doi.org/10.1023/A:1010091220143). Local rollout planner; guided/refined variants share this implementation. | [baselines](../benchmarks/baselines.py), `cem` / `hpo_cem` / `hpo_gradient`; [qualify_search](../benchmarks/qualify_search.py) | [Search qualification](baseline-implementation.md) |
| [QR-D4PG port (`hull_rl`)](../src/hedging_gym/baselines/qr_d4pg.py) | [Cao/Hull 2023 code](https://github.com/rotmanfinhub/gamma-vega-rl-hedging/tree/77dc48326da000d983b1fb750edb2177e38c75fd). Maintained PyTorch global-risk port, distinct from historical author-learner runs. | [qualify_model_free](../benchmarks/qualify_model_free.py), `--method hull_rl` | [Port versus original learner](baseline-methods.md#training-objectives-are-not-interchangeable) |
| [EX-D4PG port (`exdrl`)](../src/hedging_gym/baselines/exdrl.py) | [EX-DRL code](https://github.com/pmalekzadeh/EX-DRL/tree/f1abe99df7fa9efaa65af6b9dd416c3425c64098). Maintained PyTorch global-risk port, distinct from historical author-learner runs. | [qualify_model_free](../benchmarks/qualify_model_free.py), `--method exdrl` | [Port versus original learner](baseline-methods.md#training-objectives-are-not-interchangeable) |
| [Common AlphaZero (`alphazero`)](../src/hedging_gym/baselines/alphazero.py) | [Szehr-derived search](baseline-methods.md#method-mapping). Maintained stochastic-search and terminal-ES adaptation. | [qualify_alphazero](../benchmarks/qualify_alphazero.py), configured finite holding grid | [Qualification](baseline-implementation.md) |
| [GEPS conditioning](../src/hedging_gym/baselines/geps.py) | [published layer mechanism](geps-adaptation.md#pinned-authority). Published conditioning-layer transfer. | [compare_fast_adaptation](../benchmarks/compare_fast_adaptation.py), `--method geps` | [Source parity and limits](geps-adaptation.md) |
| [Belief-context DH](../src/hedging_gym/baselines/belief_adaptation.py) | [Belief-FB encoder source](belief-adaptation.md#pinned-source-and-actual-reuse). Published dynamics-encoder transfer. | [compare_fast_adaptation](../benchmarks/compare_fast_adaptation.py), `--method belief` | [Encoder transfer and limits](belief-adaptation.md) |
| [AMAGO](../src/hedging_gym/baselines/amago.py) | [AMAGO](https://arxiv.org/abs/2310.09971), [AMAGO-2](https://arxiv.org/abs/2411.11188); [official donor](amago-context.md). Imported official AMAGO and AMAGO-2 learners. | [qualify_amago_hedging](../benchmarks/qualify_amago_hedging.py), `--agent-type agent` or `multitask` | [Memory qualification](amago-context.md) |

Fixed delta bands, guided/refined CEM and AMAGO's `agent`/`multitask` selection
are configurations of their respective modules, not duplicate implementations.
Use `python -m benchmarks.NAME --help` for runner arguments. The
[paper configuration guide](paper-benchmarks.md) documents the Bühler Heston
and AlphaZero Heston/GBM tasks and their differences from published settings.

## Training and adaptation extensions

These additions operate on an existing policy; they do not define independent
hedgers. The runner identifies which policy receives the extension.

| Addition / implementation | Source and host method | Runner | Evidence |
|---|---|---|---|
| [Skill retrieval](../src/hedging_gym/extensions/skill_retrieval.py) | [SRSA transfer mapping](skill-retrieval.md#source-and-finance-mapping). Published retrieval-predictor transfer. | [compare_fast_adaptation](../benchmarks/compare_fast_adaptation.py), retrieval review | [Retrieval checks and limits](skill-retrieval.md) |
| [Adaptation-aware pretraining](../src/hedging_gym/extensions/meta_pretraining.py) | [first-order MAML mapping](fast-adaptation.md#pretrain-for-adaptation-or-simply-train-longer). First-order MAML principle adapted to hedging. | [compare_adapt_aware](../benchmarks/compare_adapt_aware.py), `meta` and ordinary controls | [Pretraining comparison](baseline-implementation.md) |
| [Counterfactual mode credit](../src/hedging_gym/extensions/counterfactual.py) | local [shared-continuation comparison](counterfactual-update.md). Local HPO policy extension. | [qualify_counterfactual](../benchmarks/qualify_counterfactual.py) | [Mode-credit qualification](counterfactual-update.md) |
| [Joint counterfactual updates](../src/hedging_gym/extensions/joint_counterfactual.py) | [HPO extension](joint-counterfactual.md). Local mixed-gradient extension. | [qualify_counterfactual](../benchmarks/qualify_counterfactual.py), `--joint` | [Joint-update qualification](joint-counterfactual.md) |
| [Task curriculum](../src/hedging_gym/extensions/curriculum.py) | [ACCEL](https://proceedings.mlr.press/v162/parker-holder22a.html). Local finite-task sampler inspired by ACCEL. | [qualify_curriculum](../benchmarks/qualify_curriculum.py), declared A/B mixture | [Qualification](baseline-implementation.md) |

## Use a baseline

From the checkout after `uv sync --locked`:

```bash
uv run --frozen python -m benchmarks.baselines --methods dh ntb
uv run --frozen python -m benchmarks.baselines --help
```

The default example uses 128 training paths, a separate 128-path evaluation bank,
eight updates, minibatches of 32 and hidden layers of 32, 32. Policy seed is 7;
training and evaluation seeds are 1101 and 2201. These small defaults demonstrate
execution. Use the linked qualification runners for substantive comparisons.

Policy constructors derive their schema and action sizes from the configuration:

```python
from hedging_gym import HedgingEnv, benchmark_config
from hedging_gym.baselines.deep_hedging import DirectDHPolicy

env = HedgingEnv(benchmark_config(model="gbm"))
policy = DirectDHPolicy.from_env(env, hidden=(32, 32))
env.close()
```

For direct Deep Hedging, `deep_hedging.train(bank, ...)` returns a policy and
training metadata; `deep_hedging.make_controller(policy)` connects it to
`evaluate_controller`. No-transaction bands, QR-D4PG and EX-DRL follow the same
pattern. Classical modules expose `make_controller()` without training.
Adaptive DH and full-network fine-tuning expose `make_updater(policy, ...)` for
their respective update rules. Upstream learners retain their own training APIs.

SB3, CrossQ and TQC use the optional `baselines` package extra. In a checkout:

```bash
uv sync --locked --group baselines
uv run --frozen --group baselines python -m benchmarks.qualify_sb3 --help
```

SimBaV2 and AMAGO need their pinned external donor environments; the AlphaZero
source runner needs its [pinned checkout](source-alphazero.md#run-a-bounded-qualification).
Run artifacts and checkpoints belong outside Git. `--checkpoint-dir` and
`--resume-from` in the common runner retain learner state; resuming requires the
same configuration, training bank, seed and recipe. Historical runs use their
archived source and artifacts.

## Shared interface, separate learners

Every evaluation controller follows the same small contract:

```python
controller(observed, ledger, time_index, config) -> target_holdings
```

Targets have shape `[batch, n_assets]` on the input device. Inputs are read-only;
only the environment executes trades and updates cash and holdings. Classical
controllers need no training method. A learned policy and a planner can satisfy
the same contract without sharing a network architecture or training loop.

Folder ownership follows the division of labor:

| Location | Responsibility |
|---|---|
| `interfaces.py` | Public `Controller` contract, independent of any learner. |
| `environment/` | Financial calculations, legal trades, batched stepping, complete episodes and conditional simulation. |
| `adapters/` | External environment formats: SB3 vector observations, action coordinates and reset conventions. |
| `baselines/<method>.py` | Method construction, learning or search, and conversion of its decisions to target holdings. |
| `baselines/_shared/` | Reused learning mechanics: pathwise updates, replay, networks, policy adapters and checkpoints. |
| `extensions/` | Optional changes to an existing hedger's pretraining or adaptation. |
| `evaluation.py` | Common frozen-policy evaluation and risk metrics. |

### Public interface versus internal reuse

`interfaces.py` and the environment APIs are the public integration points for
new algorithms. `baselines/_shared/` contains private implementation reused by
particular methods, not a framework every baseline must use. For example,
`_shared/d4pg.py` supports QR-D4PG and EX-D4PG; it does not participate in PPO's
training procedure. Each helper's module description identifies its purpose
and scope.

Contributors can reuse these helpers where the mechanics genuinely match, but
integrating a new algorithm requires only the public interface and environment
APIs. Method-specific behavior stays in its named module; internal reuse does
not require methods to share a network architecture or learning procedure.

`environment.rollout.run_episode` executes any controller without changing its
network mode or detaching gradients. `environment.planning.sample_continuation`
and `rollout_branches` simulate alternatives from current observations and a
read-only ledger. Candidates share fresh conditional scenarios, not the realized
evaluation future. The planner still chooses candidates and aggregates risk.
Methods needing individual transitions use `TensorHedgingEnv` or the public
financial primitives; they need not adopt an identical training loop.

PPO, CrossQ and TQC delegate learning to SB3/SB3-Contrib. SimBaV2 and AMAGO
delegate to their official learners. `adapters/sb3.py` is our connector, not a
copy of SB3. Learner-specific replay, objective relabeling and updates remain in
the method or its shared helpers, not in the format-conversion adapter.

## Add a method

Use one named module with this structure:

1. A short header naming the method, paper URL, upstream URL/version or commit,
   and implementation-document link. Identify local extensions explicitly.
2. Method setup and training, reusing the upstream learner when available.
3. A controller adapter returning common target holdings, plus checkpoint
   handling where the learner needs it. No dummy trainer for classical methods.
4. A runnable recipe under `benchmarks/` and a catalogue row documenting the
   objective, action representation and deliberate source changes.

Share helpers only when implementations genuinely reuse them. New methods must
not duplicate portfolio accounting or silently change execution rules.

For example, a differentiable policy uses the same public episode runner as
evaluation:

```python
from hedging_gym import benchmark_config
from hedging_gym.baselines.deep_hedging import DirectDHPolicy, make_controller
from hedging_gym.environment.finance import generate_market_bank
from hedging_gym.environment.rollout import run_episode

config = benchmark_config(model="gbm")
bank = generate_market_bank(config, 32, seed=7)
policy = DirectDHPolicy(config, hidden=(32, 32))
controller = make_controller(policy, evaluation=False)
result = run_episode(controller, bank)
# The learner owns its objective threshold and optimizer; autograd stays intact.
threshold = 0.0
objective = config.risk.loss(result["terminal_loss"], threshold).mean()
objective.backward()
```

A new algorithm can supply its own callable instead of a `DirectDHPolicy`.
It receives current observations, ledger, date and config and returns batched
target holdings. It need not inherit a learner base class. Read inputs without
mutating them; use `.clone()` when constructing a target by in-place changes.

If the new code only selects an embedding, changes task sampling or adds an
update to an existing policy, place that reusable mechanism in `extensions/`
and name its host policy in the recipe. A complete new hedger still gets its own
baseline entry point; helper files are not additional baselines.

Evaluate frozen policies on fresh common paths and report their training
objective, work and decision-time search budget. Terminal loss already includes
trading costs; ES-trained and MSE-trained policies are distinct comparisons.
Continuous policies need an explicit treatment of lots, minimum orders and
fixed-ticket decisions. See the [financial contract](benchmark.md) and
[implementation appendix](baseline-methods.md).
