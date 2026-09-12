# Joint counterfactual hedging updates

The earlier all-mode pilot froze continuous sizes. This experiment restores
continuous learning and its effect on subsequent trade-mode decisions, while
keeping the Heston benchmark, ledger, information and terminal ES95 unchanged.
It is an adaptation of established gradient estimators, not a new theorem.

## Sources and mechanism

- [HPO, Alvo–Russo–Kanoria](https://arxiv.org/abs/2605.14297) and its
  [author optimizer](https://github.com/MatiasAlvo/hybrid-rl/blob/e48ae86da1e8f14c93cbb56e48d87f8674228659/src/algorithms/hybrid/optimizer_wrappers/hybrid_wrapper.py)
  supply the pathwise-plus-score estimator, including derivatives through
  earlier continuous actions into later categorical probabilities.
- [Expected Policy Gradients](https://jmlr.org/papers/v21/18-012.html) motivates
  integrating discrete action credit rather than sampling one mode.
- [MPO](https://arxiv.org/abs/1806.06920) is not implemented here. Exponentiating
  a single realised market-path loss is not the same as improving a policy using
  a conditional expected action value. This experiment uses a linear unbiased
  gradient estimate, not hindsight-optimal labels or noisy softmax targets.

For fixed threshold zeta, let c(L) be the terminal Rockafellar–Uryasev cost.
The mixed gradient is the expectation of:

    pathwise_gradient c(L)
    + sum_t stop_gradient(c(L) - b(history_t)) * gradient log p(mode_t | history_t)

The history in the score remains live: its derivative includes earlier sizing
decisions. The detached baseline is independent of the selected mode. We sample
one branching date uniformly. The current `trajectory` variant retains the score
sum at **every** date on every branch; its root score appears once, without a
horizon multiplier. Each branch continues its own causal policy on the same
market path and shared future uniforms. The sampled comparator retains the same
complete trajectory scores, not an intentionally reduced update.

The all-mode objective averages each branch's pathwise-plus-score surrogate
using detached root-mode probabilities. Differentiating these weights again
would double-count the root score; detaching history would omit the HPO cross
term. Prefix scores are repeated with each branch's own terminal cost, while
suffix scores use that branch's live history. The sampled arm uses one complete
trajectory. This integrates the root mode in both gradient terms.

The earlier `sampled_date` variant remains reproducible: it estimates the score
sum from one randomly selected date times the horizon, while its pathwise term
is not multiplied. It is unbiased under the same assumptions but discards
available trajectory credit and adds time-sampling variance. The correction
restored a demonstrated HPO mechanism; it did not change the benchmark or tune
the optimizer after looking at results.

One market suffix per sampled state is sufficient for an unbiased linear
policy-gradient contribution in expectation over training paths. It is not an
accurate conditional action-value estimate, a reliable per-state optimizer,
or a monotonic policy-improvement guarantee. All-mode integration reduces
root-mode sampling noise conditional on the prefix and coupled branch table,
not necessarily gradient variance per dollar. Costs, absolute-value kinks and ties retain the
same piecewise-smooth scope as the current hybrid adapter. Lot/minimum-order
constraints are not silently rounded into this experiment.

## Development comparison

Use existing seed 7/17/29 HPO checkpoints: basic at step 3000, fixed fees at step
1000; existing 16,384 training paths and 8,192 development paths. Final paths
remain untouched. Calibrate one threshold on the incumbent's training losses.
Both joint arms freeze that threshold and critic, update mode and sizing networks
for 300 steps, use learning rate 0.0003 and base batch 64. These are the preceding
pilot settings, not a new search over hyperparameters.

At intervention date t, all-mode accounting work is B*(t+M*(T-t)). The sampled
arm uses ceil(B*(t+M*(T-t))/T) full trajectories. Report decision-ledger steps,
liquidations, wall time and backward work separately; matched accounting work is
not matched compute. All-mode autograd visits every branch, so elapsed time is
important even when accounting work is matched.

Continue full HPO from its original optimizer/RNG checkpoint for the number of
complete rollout batches needed to approximately match additional accounting
work. Its extra PPO/critic epochs, changing threshold and entropy term remain
intact. Report these as a stronger complete recipe, not as the sampled ablation.
Saved DH/basic and band/fixed results are contextual only; any positive practical
claim also needs continued strong controls at comparable added work.

The exact two-date check integrates every categorical sequence and shared-uniform
interval for both variants. It compares categorical and continuous gradients
against exact risk, and confirms that omitting the history cross term would
fail. A second identity integrates root modes conditional on a fixed prefix and
shared suffix uniforms; this verifies the full-trajectory Rao–Blackwell step.
Existing branch
cash-accounting and terminal-risk tests remain in use. Training saves optimizer,
RNG, progress and policy checkpoints; evaluation reloads disk checkpoints and
saves full trade tapes for independent cash reconstruction.

```bash
PYTHONPATH="$PWD/src:$PWD" /path/to/python -m experiments.qualify_counterfactual \
  --source-dir /path/to/qualification --output /path/to/new-joint-run \
  --preset basic --seeds 7 17 29 --updates 300 --batch-size 64 \
  --joint --score-scope trajectory --full-hpo --continue-control --device cpu
```

This is a bounded mechanism test. Saturation can still attenuate categorical
gradients; restoring sizing is not presented as a guaranteed remedy. We retain
failures and do not select benchmarks or thresholds after inspecting outcomes.

## Completed development evidence

Artifacts are outside Git. Paths below are relative to this run's
`r2-three-route-2026-09-07-plHWbD/counterfactual/` output directory.
`joint-basic` and `joint-fixed` contain the first `sampled_date` experiment
(these predate the explicit `score_scope` field). `trajectory-basic` and
`trajectory-fixed` contain the corrected complete-score experiment. Their
300-step intervention schedules give exactly the same per-seed work budgets,
so the original continued HPO/DH/band checkpoints are reused explicitly, not
retrained or relabelled. Both objective variants remain in the small adapter.

CPU timing was 0.63 seconds for 25 all-mode updates and 0.53 seconds for sampled
updates. We therefore ran basic and fixed concurrently, two Torch threads per
process, while another research lane used the GPU. Each 300-update corrected
arm took roughly eight seconds. These are development timings under contention,
not a formal CPU/GPU or inference-speed comparison.

On the original 8,192 development paths, mean ES across the three source seeds:

| Case | Source | Single-date all-mode | Trajectory all-mode | Trajectory sampled | Continued HPO | Continued DH/bands |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Basic | 0.0101533 | 0.0098127 | 0.0099376 | 0.0092462 | 0.0094918 | 0.0092767 |
| Fixed fee | 0.0133466 | 0.0123249 | 0.0120668 | 0.0123155 | 0.0129183 | 0.0134587 |

**The fixed-fee average is not a trustworthy three-seed superiority result:**
the precision diagnosis below invalidates the economic interpretation of most
seed-17 gain. Raw numbers are retained rather than deleting that seed.

The source-only 32-repetition gradient diagnostic explains one limitation.
On basic, all-mode full-gradient trace variance was 2.2–4.5 times that of
work-matched sampling. Extra independent market paths were more valuable than
extra actions on the same path. Fixed-fee behavior was heterogeneous. Per-root
Rao–Blackwellization therefore does not establish lower variance per dollar.
This is an exploratory sample-variance estimate, not a known population gradient.

## Fresh confirmation and the precision finding

Before generation we fixed 32,768 ordinary Heston paths, numeric seed 920071,
the unchanged configuration and substeps, and all three frozen trained policies.
CUDA generated/priced this bank in 4.38 seconds; frozen evaluation ran on CPU.
Both basic and fixed use those same market paths with their pre-existing costs.
No final-test paths were opened and no checkpoint was selected after evaluation.

The fresh float32 result repeats the pattern: basic all-mode is worse than the
work-matched sampled update for all seeds; fixed all-mode improves on sampled
by 0.000425, 0.000195 and 0.000026 for seeds 7, 17 and 29. Paired path-bootstrap
95% intervals exclude zero for the first two, but include zero for seed 29.
These intervals condition on the fitted policies, not on training-seed
uncertainty, and they do not cure the precision problem discovered next.

The reduction in fixed-fee seed-17 tickets triggered a numerical audit before
claiming success. At identical market marks and common uniform mode draws,
the policy still proposed almost 30 TRADE modes per episode, but only about
6.79 caused nonzero target changes in float32. About 23.20 TRADE modes produced
exactly unchanged holdings. In float64 all of those tiny changes became
nonzero trades, correctly charged by the unchanged ledger. On the first
4,096 fresh paths its ES moved from 0.012993 to 0.015281, nearly the source
policy's float64 0.015287. The sampled update has the same issue, and continued
HPO is affected less strongly. Seed 7's gain survived this precision check;
seed 29 was largely stable but its fresh advantage over sampled is uncertain.

This is **a policy/action-representation issue, not wrong cash accounting**.
The original HPO inventory donor charges fixed costs through its discrete
feature/range selection (`src/envs/inventory/hybrid_simulator.py`, function
`_calculate_fixed_ordering_costs`). Its hybrid partition isolates that cost
discontinuity. Our finance ledger appropriately charges actual trades, but an
absolute target in the purported TRADE cell can equal existing holdings through
rounding. Thus the cell does not reliably separate HOLD from nonzero execution.
Changing the ledger to waive near-zero fees would conceal rather than solve it.

### Question, explanation and next test

- **Question:** does integrating alternative trade choices improve the complete
  mixed gradient enough to beat stronger sampling and existing finance policies?
- **Evidence:** exact gradient identities pass; basic integration loses per
  work; seed 7 shows a real fixed-fee improvement; most seed-17 improvement is
  precision-sensitive rather than learned HOLD behavior.
- **Explanation:** branch work helps some discrete decisions, but does not
  automatically justify fewer independent market samples. Separately, the
  existing action representation allows machine-precision fee avoidance.
- **Next source-backed test:** make a proposed nonzero trade genuinely distinct
  from HOLD in the policy's hybrid cells, without changing feasible financial
  trades or costs; first check donor mappings and precision stability, then
  compare full HPO and counterfactual updates using the same representation.
  The source HPO fixed-cost categories isolate the discontinuity; our absolute
  target-based TRADE semantics currently do not. Any revised representation
  must be shared by full HPO and the candidate, with the existing ledger and
  no near-zero fee tolerance or selected-seed exclusion.
  A learned compute allocator or another parameter sweep is not authorized by
  these observations alone.

The lane remains scientifically open, but these results do not establish a
general superiority claim. `development-review.json`,
`fresh-confirmation/confirmation.json`, the training diagnostics and
`trajectory-fixed/precision-diagnostics.json` retain the underlying evidence.
