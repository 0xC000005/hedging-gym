# Joint counterfactual hedging updates

The [joint extension](../src/hedging_gym/extensions/joint_counterfactual.py)
updates both mode probabilities and continuous sizes in a hybrid hedger.
The value network and training-calibrated risk threshold remain frozen.
It uses the same market paths, observations, fees and cash ledger as the
[categorical-only extension](counterfactual-update.md).

## Sources and estimator

[HPO](https://arxiv.org/abs/2605.14297) and its
[pinned author optimizer](https://github.com/MatiasAlvo/hybrid-rl/blob/e48ae86da1e8f14c93cbb56e48d87f8674228659/src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py)
supply the pathwise-plus-score estimator, including derivatives through earlier
continuous actions into later mode probabilities.
[Expected Policy Gradients](https://jmlr.org/papers/v21/18-012.html) motivates
integrating discrete action credit.

For fixed threshold ζ and terminal RU cost `c(L)`, the mixed gradient combines:

```text
pathwise_gradient c(L)
+ sum_t stop_gradient(c(L) - b(history_t)) * gradient log p(mode_t | history_t)
```

Histories remain live in the score term, preserving the HPO cross term.
The baseline is detached and independent of the selected mode.

Training chooses one branching date uniformly. In `all_mode`, each root mode
receives its own causal continuation on a common market path and shared suffix
uniforms. Detached root-mode probabilities weight each complete pathwise-plus-score
surrogate. Differentiating those weights again would double-count the root score.

The default `score_scope="trajectory"` uses scores at every decision on each
branch, with the root score included once. The `sampled` comparator uses the
same full-trajectory score. The `sampled_date` ablation estimates the score sum
from one randomly selected date multiplied by the horizon; its pathwise term
is not multiplied. This adds date-sampling variance.

A single market suffix provides a linear gradient contribution in expectation.
It does not provide an accurate conditional action value or a monotonic policy
improvement guarantee. MPO and hindsight-optimal action labels are not
implemented. Enumerating root actions reduces one source of sampling noise,
but may be less efficient than sampling more independent market paths.

## Supported inputs and run

The runner supports `basic` and `operational_fixed` ES tasks with finite
continuous holdings. Trade lots, minimum-order sizes and trading at maturity
are unsupported. Use the banks and HPO snapshot layout specified in
[counterfactual setup](counterfactual-update.md#supported-inputs).

From an installed checkout, choose a new external output directory:

```bash
uv run --frozen python -m benchmarks.qualify_counterfactual \
  --source-dir /path/to/qualification --output /path/to/joint-counterfactual \
  --preset basic --seeds 7 17 29 --updates 300 --batch-size 64 \
  --joint --score-scope trajectory --full-hpo --device cpu
uv run --frozen python -m benchmarks.review_counterfactual \
  --runs /path/to/joint-counterfactual --output /path/to/joint-review.json
```

`--full-hpo` continues the complete source HPO recipe, including its critic,
PPO epochs, entropy term and threshold updates. It is a separate comparator
from the sampled estimator ablation. Optional `--continue-control` also needs
matching DH/basic or band/fixed stage checkpoints in the source directory.

For batch size B, branching date t, M modes and T decisions, all-mode ledger work
is `B * (t + M * (T - t))`. The sampled arm uses
`ceil(B * (t + M * (T - t)) / T)` complete trajectories. Matching this count
does not match backward computation, liquidation counts or elapsed time;
report them separately.

## Fixed fees and precision

Absolute target holdings can round to the current holdings even when the
policy selects a TRADE mode. The ledger then correctly charges no fee for
the zero executed trade. Float64 can resolve the same tiny change as a nonzero
trade and charge a fixed ticket, making apparent gains precision-sensitive.

Inspect executed position changes and ticket counts alongside categorical
TRADE counts, and compare float32/float64 using identical marks and random
mode draws. The policy's TRADE representation does not guarantee a distinct
nonzero execution. Adding a near-zero fee waiver would change the task and
conceal this limitation. Apply any revised action representation consistently
to HPO and its counterfactual extensions before comparing them.

## Verification and interpretation

```bash
uv run --frozen pytest -q tests/test_joint_counterfactual.py tests/test_counterfactual.py
```

Exact small-horizon checks compare continuous and categorical gradients with
enumerated risk and test the live-history cross term and root-mode integration.
Shared ledger checks verify each branch's cash accounting. These establish
estimator and implementation behavior, not improved hedging performance.

Evaluate frozen policies on fresh paths with multiple training seeds and strong
continued controls. Conditional path intervals do not capture training-seed
uncertainty; reported work counts do not establish equal runtime.
