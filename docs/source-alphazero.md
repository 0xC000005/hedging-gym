# AlphaZero source-loop qualification

This adapter runs the published [minimalHedger](https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba)
`MCTS.py`, `Trainer.py` and neural fitting code against Hedging Gym. It is a
separate source-backed comparator, not a replacement for historical adapters.
That repository accompanies Szehr's earlier MCTS paper, not the complete 2025
AlphaZero-versus-Deep-Hedging experiment.

## What is retained

PUCT selection and backups, shared search statistics, visit-probability targets,
terminal-return labels, replay across cycles, the six-layer batch-normalized
network, cross-entropy plus value MSE fitting, and reject-worse validation all
come from the donor. Fresh market outcomes are sampled, not optimized.

## What changes for the common task

- The common Heston/GBM model, portfolio, calendar, fees and ledger replace the
  source's game implementation. Its Heston branch otherwise follows a stored
  future path during search, and its GBM fresh-transition branch is incomplete.
- Observations determine network input width instead of a hardcoded 25.
  Absolute holding-grid dimensions follow the environment's instruments/limits.
- An unbounded value head replaces `tanh`. The selected terminal objective is
  not clipped to fit the original value head; see the objective choices below.
- CPU placement is explicit, avoiding the donor's CUDA-network/CPU-minibatch
  mismatch. Frozen policy evaluation is batched. The external checkout needs
  only its unused `torchvision` import removed.
- Scalar search nodes use the existing QuantLib analytic reference with the same
  model, contracts and calendar; batched evaluation retains the tensor pricer.
  Their marks are checked for agreement at ordinary, zero/near-zero variance
  and late-date states. Search omits unused nonterminal liability prices.
  Exactly zero variance uses the tensor formula because QuantLib's model
  constructor requires positive variance; market states are never floored.

## Objective choices

The current comparison uses **Heston or GBM with terminal hedging MSE**, not a
new trinomial environment. Market dynamics, instruments, fees and cash accounting
are unchanged when selecting the objective.

- `risk.objective: "mse"` in JSON minimizes `E[L²]`, where `L` is terminal
  cost-inclusive hedging loss. Reward is `−1 − (L/s)²` for fixed monetary scale s.
  The constant avoids the source's zero/nonterminal flag; scaling does not change
  the optimum. Both gains and losses enter the square. There is no clipping,
  centering or extra transaction-cost penalty, and no ζ calibration is needed.
- `risk.objective: "es"` retains the earlier experiment: reward
  `−1 − max(L−ζ, 0)/((1−α)s)` minimizes a fixed-ζ Rockafellar–Uryasev objective.
  ζ is calibrated on separate initial-policy paths and frozen through the block.
  Longer ES training requires threshold updates and consistent replay labels.

The common evaluator reports raw MSE, RMSE and ES from complete fresh terminal
losses. Under MSE training, ES95 is a diagnostic, not the optimized objective.
The CLI `--objective` can explicitly override the JSON choice. Runs using a
common `--market` preset default to MSE; runs using `--config` follow its risk
configuration. The scalar, vector and tensor Gym interfaces also read that
configuration and return terminal negative squared loss for MSE. The source
adapter adds only its documented constant shift and monetary scaling.
This objective switch is an alternative, not an MSE-plus-ES weighted loss.
Existing ES checkpoints keep their original classification. Any competitive
comparison must also train Deep Hedging for MSE on the same financial task.

Using these market parameters and squared loss is a common-benchmark adaptation,
not an exact paper reproduction: our portfolio and execution settings differ,
and we do not use the donor's clipped reward.

The source's rounded transposition keys are retained for nonterminal states
(variance: 3 decimals; spot, cash, holdings: 2). Financial values themselves
remain unrounded. Terminal cache keys retain exact loss. Approximate state
aggregation and the source replay dictionary's overwriting of repeated states
are limitations to assess, not new features. Search depth is measured.
Minimum-order constraints require action masking and are not supported here.
Lot-aligned grids are supported when every target is reachable from the initial
inventory; subsequent grid-to-grid trades then remain aligned.

## Match the source market, book and calendar by configuration

`--config` loads a complete financial JSON through the Gym's existing
`config_from_dict`; it replaces defaults rather than merging leftover instruments
or costs. It is mutually exclusive with the common-preset `--market` option.
The same configuration object works with `HedgingVectorEnv` and other learners.

| Configuration | Decisions | Year clock | Traded instruments | Costs |
|---|---:|---:|---|---|
| [Heston stock-only](../experiments/configs/szehr-heston-stock.json) | 60 | 365 | Stock and cash | Zero |
| [GBM stock-only](../experiments/configs/szehr-gbm-stock.json) | 30 | 365 | Stock and cash | Zero |

Both sell one ATM call, omit all hedge options (`hedges: []`), and allow stock
holdings between −1 and 1. Select `--grid-points 21` for the donor's action grid.
The settings follow the pinned donor's
[Heston](https://github.com/plan64/minimalHedger_AlphaZero/blob/3111c378fcd17e45f94d2fc668a3aa117126ecba/hedger_TV/hedgerGame_TV_heston.py)
and [GBM](https://github.com/plan64/minimalHedger_AlphaZero/blob/3111c378fcd17e45f94d2fc668a3aa117126ecba/hedger_TV/hedgerGame_TV_pureGBMPaths.py)
classes. The objective is its own config component; action-grid density and
training budget remain learner arguments. A model name does not choose
instruments or a loss.

These files match those financial components, not every source convention.
The donor Heston discretization is Euler; the supplied Gym configuration selects
QE-M, an approximate scheme with a conditional stock martingale correction. The
donor Heston initial cash is hardcoded to 0.021; Gym retains its model-price initial
capital. The original reward is clipped and has a cost-state offset even at zero
fees; `--objective mse` deliberately uses actual, unclipped terminal squared loss.
These differences must be stated in any results comparison.

To test arithmetic against the actual donor code, run:

```sh
uv run --frozen --group baselines --with scipy python -m experiments.check_source_alphazero_parity \
  --donor /path/to/minimalHedger_AlphaZero --output /path/to/new-parity.json
```

This replays donor-generated paths and exact donor action quantities through
the shared ledger. It reports byte equality and maximum numerical differences
for cash and terminal loss. Initial capital is harmonized only in this audit;
both original premiums and their difference are recorded. It also reports the
original clipped reward separately from raw MSE. Scientific search continues
to draw fresh conditional transitions.

Matching a configuration does not make different random generators, integration
schemes or training implementations byte-identical. Fixed-input arithmetic can
be checked exactly; stochastic performance must be compared on fresh paths.

## Run a bounded qualification

Install the baseline dependency group with `uv sync --locked --group baselines`.
Clone the pinned donor outside this repository and remove its unused torchvision
import from `hedger_TV/neuralNet/hedgerNeuralNet_simpleFF.py`. Then run:

```sh
uv run --frozen --group baselines python -m experiments.qualify_source_alphazero \
  --donor /path/to/minimalHedger_AlphaZero --output /path/to/new-run \
  --market heston --objective mse
```

Defaults run one cycle of eight full 30-date Heston episodes, 25 simulations per
decision and two supervised epochs. This checks execution, not convergence.
Use `--market gbm` for the other existing market core. To repeat the historical
tail-loss qualification, select `--objective es` explicitly.

For the source-sized Heston market and stock-only book:

```sh
uv run --frozen --group baselines python -m experiments.qualify_source_alphazero \
  --donor /path/to/minimalHedger_AlphaZero --output /path/to/new-run \
  --config experiments/configs/szehr-heston-stock.json --grid-points 21 --objective mse
```

Logs show self-play/refit/validation progress; artifacts retain objective, configuration,
source commit/diff, initial/candidate/selected weights, replay examples and fresh
evaluation tapes. Checkpoint loading is verified exactly; this is not a claim
of interruption-exact optimizer resumption. No final test data selects a model.
