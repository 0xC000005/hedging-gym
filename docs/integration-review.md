# Integration and cleanup — 2026-09-12

This is a code/evidence preservation checkpoint, not a new scientific result.
The user approved the architecture, one squash integration commit above
`70faa144`, private push and removal of the audited stale worktrees and branches.
The annotated tag
`evidence-2026-09-12-hedging-closeout` retains the assembly ancestry below.
Source branches and linked checkouts are disposable only after main and the
evidence tag are verified remotely. External run artifacts are not cleanup targets.

## Retained branch coverage

Paths in this table are the original paths at
`evidence-2026-09-12-hedging-closeout`; the current organization is listed below.

| Original branch tip | Code retained at the evidence tag |
|---|---|
| `feat/baseline-suite` · `433b5ff` | Common financial core, configured paper tasks, baseline suite and separate source-loop AlphaZero bridge |
| `feat/advanced-offpolicy` · `00afb506` | `methods/off_policy.py`, qualification runner and replay/continuation checks |
| `feat/simbav2-baseline` · `ad7fe1d3` | `methods/simbav2.py`, optional official-donor runner/tests |
| `feat/hull-ddpg-qualification` · `70e32b3c` | `methods/hull_ddpg.py`, distinct 2021 two-moment learner and runner |
| `feat/r2-fast-adaptation` · `05fc719f` | Fast-adaptation comparisons, `methods/meta_pretraining.py`, attribution and search diagnostics |
| `feat/r2-geps-adaptation` · `c64b8c31` | `methods/geps.py`, source-parity tests and attribution notes |
| `feat/r2-srsa-adaptation` · `689498e9` | `methods/skill_retrieval.py` and transfer-risk tests |
| `feat/r2-belief-adaptation` · `873b67c3` | `methods/belief_adaptation.py`, encoder tests, source/license notes |
| `feat/r2-counterfactual-update` · `549d17a2` | `methods/counterfactual.py` and qualification/review runners; already ancestor of three-route work |
| `feat/r2-three-route-adaptation` · `54e21a42` | `methods/amago_adapter.py`, optional qualification runners/tests |
| `feat/r2-three-route-counterfactual` · `e6878090` | `methods/joint_counterfactual.py`, matched diagnosis and confirmation runners |
| `feat/r2-three-route-curriculum` · `d95f3b72` | `methods/curriculum.py`, uniform/regret/pooled-RU controls and qualification runner |

All tips are ancestors of the assembly tag. Overlapping GEPS/retrieval/belief
patches were reconciled against their later fast-adaptation versions; no second
active copy was imported. Earlier implementations remain in tag ancestry rather
than alternate active modules. No unique branch is left outside that ancestry.
External donor checkouts, environments, checkpoints, raw tapes and dated results
are deliberately not vendored. No old Thesis-Experiments core was imported.

## Current configuration and archive boundary

Financial simulation equations are unchanged from the preserved baseline-suite
tip; dictionary loading now uses current defaults for omitted fields. Learners
live in `src/hedging_gym/baselines/`; reusable additions live in `extensions/`;
external-library environment translation lives in `adapters/`. The financial core lives in
`src/hedging_gym/environment/`, and `src/hedging_gym/evaluation.py` remains the
common evaluator. Both subpackages install together. Checkout runners and paper
configuration files live in `benchmarks/` and use that one financial ledger.
Maintained runners use current QE-M defaults and explicit configured tasks.
Historical plain-QE scores are archived evidence, not current-task qualification.
Historical replay uses matching archived source and artifacts; current code
does not maintain old-checkpoint compatibility. The earlier compatibility helpers,
legacy configuration pin and migration tests were removed at the user's request.
The evidence tag remains unchanged and preserves that earlier source if needed.

Current checkpoints must match explicit configuration. Retrieval excludes
categorical model/scheme values from numeric features and rejects scheme mismatch.
Experimental contracts are explicit: embedding and SimBa policies need finite
bounds; meta/curriculum/CrossQ/TQC/SimBa training is ES-specific; AMAGO and
counterfactual branches reject maturity-date trading. No new algorithm recipe
was introduced. Some optional donor runtimes and historical runner duplication
remain; this closeout does not attempt a framework or dependency overhaul.

See [current qualification status](baseline-implementation.md) for completed,
weak and negative results. Adaptive DH is implemented; conditional Deep Bellman
Hedging is not. Stronger ordinary pretraining is a stronger baseline, not novelty.

## Organization checklist

- [x] Move the financial core and baseline modules into the two package areas,
  keeping the environment independent of learner imports.
- [x] Move checkout runners and configuration files to `benchmarks/` and update
  active imports, commands and tests.
- [x] Update the README, baseline guide and package metadata for one installed
  package; preserve archived source and dated results.
- [x] Separate named methods into their own public modules. DH and learned
  bands share `_shared/pathwise.py`; their policies live in `deep_hedging.py` and
  `no_transaction_band.py`. Shared machinery has private, responsibility-based
  names rather than mutually exclusive algorithm-family categories.
- [x] Verify the locked environment, package contents, focused checks and final
  diff before reporting the review branch ready.
- [x] Architecture approved; fold the reviewed cleanup into one local squash
  integration commit above the unchanged main.
- [x] Obtain approval for main integration, private push and audited cleanup.

## Method-level cleanup checklist

### Public interface and responsibility cleanup

- [x] Put the public controller contract outside baseline implementation;
  separate external-library environment adapters from learner code.
- [x] Keep named methods in `baselines/`, shared algorithm mechanics in
  `baselines/_shared/`, and composable research additions in `extensions/`.
- [x] Expose shared differentiable episodes and conditional branch execution
  without moving learner updates or changing financial equations.
- [x] Update contributor examples and check imports, gradients, seeded behavior
  and packaging; obtain architecture approval before the final squash.

- [x] Give each named baseline its own public module and move shared machinery
  into private helpers; preserve algorithms and configured objectives.
- [x] Formalize the existing batched controller interface and retain upstream
  learners behind method-specific adapters.
- [x] Update runner/test imports, source headers and the flat method catalogue.
- [x] Check numerical behavior, focused tests and installed package imports.
- [x] Present the final architecture for review before staging or squashing.

The public method map is [Baselines](baselines.md). `interfaces.Controller`
formalizes the existing read-only inputs and batched target-holdings output;
it introduces no learner superclass. Public policy adapters and adaptation
factories reuse private implementation helpers. PPO/CrossQ/TQC still train via
SB3/SB3-Contrib; QR-D4PG and EX-DRL expose distinct critics with shared updates.
Their source status and financial changes are explicit in each module header.

Grouped source modules were replaced by the named modules and shared helpers;
historical implementations remain recoverable through the evidence tag.
An obsolete D4PG checkpoint-default fallback was removed. Current-format
save/resume remains tested; historical checkpoints use historical source.

## Validation scope

The integrated source received financial/API tests and optional donor checks.
After removing migration support, only the affected current-config continuation,
feature and supported-contract tests were rerun: **31 passed, 1 optional SimBa
donor skip**. Compatibility-only tests and
omitted-field mutations are removed; ordinary same-config resume tests remain.
Seven new integration test functions cover real feature/objective/interface
constraints, not historical checkpoint formats. No old-checkpoint probes or new
scientific runs are part of the simplified maintained version.

Tests establish implementation behavior, not convergence or publication readiness.

After the package and baseline cleanup, the existing suite passed **170 tests,
3 skipped**. Lock/build checks, installed-wheel DH/band evaluation checks and
financial/source parity checks passed. No tests or scientific runs were added.

After the method-level split, the same suite again passed **170 tests, 3 skipped**
(optional donor checks), with the four existing Gym/SB3 warnings. Focused
adaptation checks also passed. Import/undefined-name checks, lock validation,
wheel/sdist construction and local documentation links passed. An installed-wheel
check outside the checkout imported 22 method modules and exercised public
training, controller and adaptation factories. Optional upstream adapters were
checked in the locked baseline environment, not claimed present in the core wheel.

A separate before/after check used float64 GBM, three trading decisions,
24 training paths, 12 held-out paths, and two updates for DH, bands, QR-D4PG and
EX-DRL. Initial/trained parameters, optimizer and active replay state, critic
outputs/gradients, decisions and losses matched exactly: 310 tensors and 918
scalar fields. Classical delta/gamma/variance decisions also matched. Timing,
prose metadata and unused replay capacity were excluded. All 12 financial-core
and paper-configuration source files were byte-identical to the pre-split snapshot.
These are refactor-preservation checks, not new performance results.

### Public interface extraction

The public contract now lives at `hedging_gym.interfaces.Controller`.
`environment.rollout.run_episode` supports plain controllers and differentiable
policies without changing model mode. Evaluation and pathwise training use this
same runner. `environment.planning` owns conditional scenarios and branch
execution; CEM retains proposals, elite selection and risk scoring. The donor
AlphaZero scalar pricing adapter now calls `environment.pricing`.

Branch execution copies action buffers and uses ledger precision, matching
ordinary environment steps. One new parametrized test checks branch/episode
losses and gradients, buffer reuse, mixed precision, and maturity-date trading.
Existing gradient tests also exercise the public episode API directly.

The environment adapter for SB3 is separate from its learner-specific replay
and updates. Shared implementation moved under `baselines/_shared/`; five
composable additions moved to `extensions/`. Complete GEPS, belief-context and
AMAGO method implementations remain in `baselines/`. No learner framework or
checkpoint compatibility aliases were introduced. In particular, historical
SB3 replay pickles require their archived defining module paths.

A bounded integration correction makes D4PG/Hull collectors and PPO rollout
sizing count `n_decisions`, including an optional maturity-date action, rather
than assume one decision per market step. Basic configuration counts are unchanged.

Seeded DH/band/QR/EX training and classical decisions still match the pre-change
snapshot exactly (310 tensors, 918 scalar fields). CEM's plain, guided and
gradient-refined variants also match exactly on both GBM and Heston, including
the action tapes, losses and work counts. Financial equations and supplied paper
configuration files were not edited. Temporary source snapshots and check scripts
are under `/tmp/hedging-interface-cleanup-1iPt97/`.

Final verification: **172 passed, 3 optional donor skips**, with the four
existing Gym/SB3 warnings. Contributor examples, CUDA episode/backpropagation,
maturity-action collection/training, import checks and local documentation links
passed. Wheel/sdist construction and an outside-checkout installed-wheel check
passed, importing 17 baseline and 5 extension modules and exercising training,
evaluation, full-network updating and embedding updating. The wheel includes the
new interfaces/adapters/environment modules and excludes obsolete module paths.

The approved cleanup uses one squash integration commit and the unchanged
evidence tag. Integrate and push both before removing the source worktrees and
branches. The pre-removal audit found clean linked checkouts, no active jobs
using them and only disposable environments/caches inside them. Retain the
primary checkout and every surrounding run directory, checkpoint and raw result.
