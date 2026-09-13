# Baseline mechanisms and financial adaptations

This guide maps the implemented methods to their sources and explains changes
needed for the shared financial task. Use the [baseline catalogue](baselines.md)
to find method modules and runnable benchmarks, and the
[reproducibility guide](baseline-implementation.md) for setup and checks.

## Shared financial task

The default `benchmark_config()` uses Heston with stock, one longer-dated European
call and cash to hedge a short European call. There are 30 daily decisions, proportional
trading costs and holding limits; no fixed fees, lots or regime changes. This
follows the simulated, self-financing design of Deep Hedging, but using a
tradable call rather than the original paper's variance-swap hedge is a disclosed
benchmark choice. [Paper configurations](paper-benchmarks.md) select other books,
calendars and objectives; all comparisons must identify the configuration used.

All controllers receive the same causal observations and start with the same
capital. Targets enter the same ledger, including trading costs and final
liquidation. Discretized methods must disclose their available holding grid;
continuous methods are not silently rounded to that grid. Market paths used for
search are sampled conditionally, not read from the realized evaluation future.

The default risk objective is upper-tail ES95 of terminal loss. The shared
evaluator also reports MSE, mean loss, transaction costs and turnover. Evaluation
metrics do not change a learner's training objective. Report training work and
decision-time work separately, and use multiple training seeds to measure
optimizer variability; resampling evaluation paths alone does not do so.

## Method mapping

References are in [related work](related-work.md); method headers contain pinned
upstream source links and the implementation notes. A source mechanism ported
to this task is not a reproduction of the source paper's full experiment.
QR-D4PG, EX-D4PG and the common AlphaZero implementation are runnable adaptations;
the available implementation checks do not establish source-paper reproduction,
convergence or strong-baseline performance.

| Method | Mechanism retained | Common-task change and implementation boundary |
|---|---|---|
| Delta | Current-state delta determines stock holdings | Uses the shared pricer and configured holding limits; does not optimize ES. A fixed delta band is an optional configuration. |
| Delta-gamma | Match liability delta and gamma using stock and an option | Uses the shared pricer and configured holding limits; does not optimize ES. |
| Delta-variance | Match liability delta and variance sensitivity using stock and a variance-sensitive claim | Uses the configured hedge instruments and shared sensitivities; does not optimize ES. |
| Deep Hedging | Differentiate terminal portfolio loss through causal trading decisions | PyTorch policy and common accounting; configured target holdings and risk objective. ES training also fits its threshold. Market simulation does not expose future observations to the policy. |
| No-transaction bands | Learn an interval of acceptable holdings; trade to its boundary when outside | One interval per instrument, with learned centers and widths. This is a band architecture, not ordinary delta hedging. |
| Deep Bellman Hedging | Actor-critic value iteration: the critic learns the risk-adjusted excess value of a book; the actor and an OCE shift network maximize the one-step monetary-utility Bellman target | PyTorch implementation of the paper's equations on the Deep Hedging actor. Rewards are one-step marked-wealth changes from the shared ledger; the reward enters the utility as in the paper's definition; cash is not a network input; training books are randomized holdings on simulated paths rather than tabulated history. The CVaR utility equals the ES threshold form pointwise, but its nested recursion is a different terminal objective (below). |
| Cao/Hull 2021 DDPG | Prioritized replay, target actor/critics, first- and second-moment Bellman targets, actor improvement through critics | Joint stock/call actions replace stock-only trading. The source mean-plus-standard-deviation objective is distinct from ES. Any source-code versus paper-equation corrections are disclosed separately. |
| QR-D4PG (`qr_d4pg.py`) | Quantile critic and deterministic actor gradients through estimated terminal risk | Local PyTorch quantile critic with the attributed shared learner, uniform replay, Polyak targets and a global terminal-ES objective. Joint holdings replace the source option action and automatic delta hedge. |
| EX-D4PG (`exdrl.py`) | Quantile critic with a fitted generalized Pareto tail in targets and actor improvement | Maintained PyTorch port with the same global-risk collector; inverse-CDF quadrature and analytic tail expectations replace sampled tail integration. |
| Adaptive Deep Hedging | Shared network and source-task embeddings; adapt a new embedding with shared weights frozen | Paper-mechanism implementation with common observations, configured holdings and terminal risk objective. Full-network fine-tuning is a separate control. |
| Full-network fine-tuning | Update all pretrained DH network weights on a new task | Uses the same financial task and updater accounting as embedding adaptation, with a different set of trainable parameters. |
| Common AlphaZero | Network-guided search, visit-distribution policy targets and realized-return value targets | `alphazero.py` uses stochastic financial branches, terminal ES and value reanalysis. Its controller performs search over a configured finite holding grid. |
| AlphaZero source loop | Import Szehr's search, training loop and network fitting | `source_alphazero.py` supplies the financial game boundary and deploys policy argmax at evaluation. Objective and state-aggregation changes are in the [source guide](source-alphazero.md). |
| SB3 PPO | The installed PPO learner, without changing its update equations | A thin batched environment maps normalized actions to holdings and supplies the declared complete-episode reward. This is an independent RL control, not an implementation of the Hull or EX papers. |
| CrossQ | SB3-Contrib's batch-normalized SAC-style learner without target networks | Continuous stock/call holdings and a terminal global-risk reward. Replay rewards must use the current training-only ES threshold, not the threshold stored when a path was collected. |
| TQC | SB3-Contrib's truncated ensemble of continuous quantile critics | Same financial interface and global-risk reward as CrossQ. Quantile truncation addresses critic overestimation; the reward mapping, not truncation, supplies the ES objective. |
| SimBaV2 | Official JAX learner: hyperspherical normalization and SAC-style distributional value learning | Thin observation/action/reward bridge, with the same financial ledger and current-threshold replay treatment. No new market model or discretized action grid. |
| Hybrid policy optimization | Discrete score-function gradients plus differentiable continuous sizing | HOLD/TRADE modes and shared cost-inclusive terminal risk; subsequent categorical PPO updates use detached histories. No straight-through derivative is substituted for the discrete choice. |
| CEM / guided rollout search | Sample candidates, evaluate conditional rollouts, refit to elites | Root-action improvement with frozen-policy feedback is a rollout-planning baseline, not full AlphaZero. Gradient refinement and learned guidance are separately costed ablations. |
| GEPS-conditioned DH | Shared low-rank layers modulated by a task context | Transfers the published conditioning equations into a hedging policy; shared weights freeze during context adaptation. See the [source mapping](geps-adaptation.md). |
| Belief-context DH | Infer a context from earlier state/action/next-state transitions | Transfers the Belief-FB dynamics encoder into DH, without the source Forward-Backward learner. Market parameters remain observed. See the [encoder guide](belief-adaptation.md). |
| AMAGO | Official sequence-based actor/critic learning over completed books | Imports AMAGO or AMAGO-2; the financial wrapper preserves memory across books and uses a fixed source-calibrated RU threshold. This is not joint ES-threshold optimization. See the [sequence guide](amago-context.md). |

The [training extensions](baselines.md#training-and-adaptation-extensions) operate
on these policies. Retrieval selects a source context; adaptation-aware
pretraining changes the initialization; counterfactual updates change HPO's
gradient estimator; and curriculum changes task sampling. Their recipes must
identify the host policy and charge any calibration or extra simulation work.

## Training objectives are not interchangeable

For ES training, Deep Hedging and the common global-risk learners use the
Rockafellar–Uryasev loss

$$
\zeta + \frac{(L-\zeta)^+}{1-\alpha},
$$

where $L$ is terminal cost-inclusive loss and $\zeta$ is one episode-global
threshold. Pathwise trainers fit the threshold jointly with the policy; other
learners calibrate it between training phases or hold it fixed, as their recipes
specify. The shared evaluator computes empirical terminal ES after pooling
complete paths, not an average of minibatch ES estimates.

Where a method supports `RiskConfig(objective="mse")`, its objective is the
sample mean of $L^2$ and no ES threshold is needed. Support is method-specific;
changing the evaluation metric does not convert an ES-trained policy into an
MSE-trained comparator. Trading costs are already included in $L$.

`RiskConfig(objective="entropy", risk_aversion=\lambda)` selects the entropic
risk $\frac1\lambda \log \mathbb{E}[e^{\lambda L}]$, the exponential-utility
certainty equivalent (Föllmer and Schied). Pathwise trainers use its optimized
certainty equivalent form $\zeta + \mathbb{E}[(e^{\lambda(L-\zeta)}-1)/\lambda]$
(Ben-Tal and Teboulle 2007) with the same jointly fitted threshold as ES; the
minimum over $\zeta$ is the entropic risk itself. The shared evaluator reports
the pooled closed form, checked in `tests/test_config.py` against pfhedge's
`EntropicRiskMeasure` and the Gaussian identity $\mu + \lambda\sigma^2/2$. The
entropic risk is the only optimized certainty equivalent besides the mean that
is time-consistent, which is what makes it the objective for comparing terminal
pathwise training with Bellman-style recursive training.

Deep Bellman Hedging optimizes a *nested* monetary utility: each date applies
the utility to its reward plus the next date's value. Nested risk measures are
time-consistent by construction, but the nested CVaR composition is not the
evaluator's terminal ES; only the entropic utility and the mean make nested and
terminal objectives coincide (the paper's Theorem 2 at zero rates). Report
`dbh` with the CVaR utility on an ES task as a different objective, and use
`objective="entropy"` for a like-for-like comparison with pathwise Deep Hedging.

The paper's one-step estimator is weak for hedging: a one-day target carries one
day of variance against one daily move of noise, so the policy's inter-temporal
cost-versus-risk trade-off must come from the critic's derivative in holdings,
which semi-gradient regression on noisy targets does not resolve. Three options
of `train_deep_bellman` change the estimator, not the objective: `steps=n` uses
the paper's n-step operator T_n (rewards of n decisions by the current policy
with pathwise gradients, then the continuation value; the horizon gives the
pathwise Deep Hedging objective per sampled state), `scenarios=K` averages the
one-step target over K fresh conditional continuations instead of the bank's
next day, and `aggregate="entropic"` replaces the learned OCE shift by the
closed-form entropic certainty equivalent of those scenarios. Declare the
values used; `steps=1, scenarios=1` is the published scheme. The paper's own
remark on multiple time steps notes that the fixed point of $T_n$ differs from
that of $T$ unless the utility is time-consistent: with the entropic utility or
the mean the two coincide and `steps=n` changes only the estimator; with the
CVaR utility `steps>1` optimizes a different nested objective and must be
reported as such. The authors' numerical companion (Murray et al., ICAIF 2022)
does not run the bare scheme either: it adds a Polyak-averaged target critic,
an actor skip connection over the Black-Scholes delta, a critic residual over
the book value, on-policy state collection and exponential-form losses. None
of these is implemented here.

The original 2021 actor instead uses critics for conditional first and second
moments. The original 2023 and EX learners use conditional distributional risk.
Reporting all their policies' terminal ES does **not** make their training
objectives identical. Preserve original-objective rows and identify any separate
common-objective variant; do not claim an algorithm is inherently inferior from
a comparison that changes both its objective and implementation.

The `qr_d4pg.py` and `exdrl.py` modules are global-risk PyTorch adaptations. The runner
key `hull_rl` selects QR-D4PG; `hull_ddpg.py` implements the separate
2021 two-moment method. Their configured actions, replay and objectives differ
from the original TensorFlow QR-D4PG and EX-DRL learners. Results obtained with
one implementation do not validate another.

The [Hull DDPG module](../src/hedging_gym/baselines/hull_ddpg.py) documents its
source corrections: pointwise nonnegative variance, target-weight copying
after initialization, and sampling from every occupied replay slot. Its actor
and critic normalization and minibatch updates retain the stated source
conventions. These details matter when comparing fixed-input behavior to the donor.

## What makes an implementation defensible

- Map the cited algorithm to its actual update, loss, action representation and
  training recipe. Native code, a paper-mechanism reimplementation and a
  common-task adaptation receive different labels.
- For changed calculations, compare fixed-input outputs, targets or gradients
  with the source. Check saved-policy reloads and independently reconstruct the
  executed cash ledger. A passing API test alone is insufficient.
- Train on declared data/budgets, retain learning curves and checkpoints, and
  investigate unexplained collapse before using it as evidence against a
  method. Source recipes may differ in budget; report those differences.
- Freeze policies before fresh final evaluation. Compare identical paths and
  initial capital; report seed variation and complete compute. Preserve poor
  results without attributing them to the broader algorithm prematurely.

Batching, GPU execution and numerical streaming are implementation choices.
Their checks establish the numerical scope of the change; equal seeds alone do
not prove identical CPU/GPU random samples or bit-identical optimizer histories.
