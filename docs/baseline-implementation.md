# Baseline reproducibility

Hedging Gym provides executable methods, paper configurations and checks of the
financial environment. It does not yet publish a cross-method leaderboard or
trained checkpoint collection. An implementation check, an author's native
result and a comparison on the common benchmark answer different questions.

Start with the [method catalogue](baselines.md) for a runner and the
[source mapping](baseline-methods.md) for changes to each published method.
In particular, the QR-D4PG, EX-DRL and common AlphaZero adaptations are not
certified reproductions or qualified performance references for their papers.

## Run and save a small comparison

From a checkout after `uv sync --locked`, run:

```bash
uv run --frozen python -m benchmarks.baselines \
  --model heston --preset basic --methods dh ntb \
  --train-paths 64 --eval-paths 64 --updates 2 --batch-size 32 \
  --seed 7 --train-seed 1101 --eval-seed 2201 \
  --checkpoint-every 1 \
  --checkpoint-dir ../hedging-gym-runs/example/training \
  --output-dir ../hedging-gym-runs/example/evaluation
```

Use a fresh output directory for each run. This example checks training, saving
and evaluation with classical controls. Heston prices and classical sensitivities
can take a few minutes on CPU; progress is printed during the run.
Two updates and 64 evaluation paths are
far too small to compare hedge quality. The common runner uses the configured
terminal expected-shortfall objective; paper-specific runners may use MSE or a
different source objective.

The example writes:

| Artifact | Contents |
|---|---|
| `evaluation/comparison.json` | Financial configuration, command arguments, seeds, training metadata, evaluation metrics and timings |
| `evaluation/heldout-bank.pt` | Evaluation market paths, configuration and simulation seed |
| `evaluation/evaluation-tapes.pt` | Per-policy losses and trade/accounting records |
| `evaluation/dh-checkpoint.pt`, `ntb-checkpoint.pt` | Policies at the end of initial training and their metadata |
| `training/<method>-seed7/latest.pt` | Periodic training state for supported resumption |

The three seed arguments have separate roles: `--seed` initializes the policy
and minibatch generator; `--train-seed` generates training paths; `--eval-seed`
generates the separate evaluation bank. Equal training and evaluation seeds are
rejected. Record the source revision with `git rev-parse HEAD` and retain the
lockfile; a seed alone does not specify the software, device or numerical scheme.

The common runner can regenerate its training bank from the recorded recipe or
load a saved bank with `--train-bank`. Its periodic training snapshots and final
evaluation checkpoints serve different purposes. Resume one supported method
with `--resume-from`, using the original configuration, bank, seed and recipe.
Only load trusted checkpoint files. See each external learner's guide for its
own checkpoint format and resume support.

## Prepare a meaningful comparison

1. **Freeze the financial task.** Record the model, numerical scheme, trading
   calendar, liability, hedge instruments, initial capital, action constraints,
   costs and objective. Use the same information and evaluator for every method.
2. **Specify each implementation.** Record its source and deliberate changes.
   Match the paper's objective when claiming reproduction. Common-environment
   adaptations should be named as such.
3. **Train across declared seeds.** Use enough data and updates to inspect
   learning behavior. Select settings and checkpoints on development data, not
   on the final evaluation paths.
4. **Evaluate on fresh shared paths.** Freeze policies, preserve paired loss
   tapes and report uncertainty across paths and independently trained models.
   Report training, simulation and decision-time computation separately.
5. **Publish a complete recipe.** Include all declared seeds and outcomes,
   configuration, commands, source revision and accessible checkpoint artifacts
   alongside a reviewed results table.

For adaptation, also record which parameters are revealed and which weights or
embeddings may update. Market changes and execution overlays are independent
configuration choices. See the [adaptation guide](fast-adaptation.md).

## Validate the environment and a new method

The [validation tutorial](validation.md) checks QuantLib price references,
fixed-input simulation transitions, independent cash reconstruction and the
Gymnasium interface. The [paper configurations](paper-benchmarks.md) explain
what is matched to published Heston/GBM studies and what differs.

For a learner, additionally inspect whether training improves its own objective,
whether saved-policy evaluation reloads consistently, and whether its trade
tapes reconcile through the common ledger. Critic and search methods also need
continuation-value checks against fresh simulation. Passing these checks is a
basis for comparison, not a guarantee that a method will win.

SB3-based runners use `uv sync --locked --group baselines`. SimBaV2 and AMAGO
require separate pinned author environments, and the source AlphaZero loop
requires an external checkout. Their module headers, runners' `--help` output
and linked guides specify the dependencies. The default installation does not
download or train these optional methods.
