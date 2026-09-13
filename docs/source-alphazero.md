# AlphaZero source-loop adapter

[SourceHedgingGame](../src/hedging_gym/baselines/source_alphazero.py) connects
the published donor MCTS, trainer and supervised fitting code to Hedging Gym.
Evaluation deploys the learned policy's argmax; it does not run a fresh search.

## Source and retained behavior

The source is [Szehr's MCTS paper](https://arxiv.org/abs/2102.06274) and
[minimalHedger_AlphaZero](https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba),
pinned at `3111c378fcd17e45f94d2fc668a3aa117126ecba`. The adapter uses
`MCTS.py`, `Trainer.py` and the six-layer batch-normalized network/fitting code.
This source accompanies the earlier MCTS paper, not the complete later
AlphaZero-versus-Deep-Hedging experiment.

PUCT selection and backups, visit-probability targets, terminal labels, replay
across cycles, cross-entropy/value-MSE fitting and reject-worse validation remain
donor mechanisms. Nonterminal transposition keys round variance to three
decimals and spot/cash/holdings to two; financial quantities themselves remain
unrounded. Terminal keys retain exact losses. Approximate state aggregation and
overwriting repeated replay keys remain limitations.

## Financial adaptation

The common market, configured book, calendar, execution rules and cash ledger
replace the donor game. Search samples fresh conditional market moves instead
of following the donor Heston game's stored future path. Observation and action
dimensions follow the configured instruments and limits.

The adapter calls the external network's forward method and captures its raw
value-head output before `tanh`, so terminal objectives need no clipping.
CPU placement is explicit. Scalar search marks use the environment's
QuantLib backend; batched evaluation uses tensor pricing. At exactly zero
variance, scalar marking uses the tensor formula because QuantLib requires
positive initial variance. No market-state floor is introduced.

The external donor checkout needs its unused
`from torchvision import datasets, transforms` import removed from
`hedger_TV/neuralNet/hedgerNeuralNet_simpleFF.py`. Keep that modification explicit
and verify the donor commit before running. The runner records actual HEAD and
the unstaged diff; it does not enforce the pin or capture staged/untracked edits.

## Objectives and supported configuration

Let L be complete terminal hedging loss, including transaction costs, and s a
fixed monetary scale:

| Configuration | Donor-game terminal reward | Meaning |
|---|---|---|
| `risk.objective="mse"` | `-1 - (L/s)^2` | Uncentered terminal MSE |
| `risk.objective="es"` | `-1 - max(L-ζ,0)/((1-α)s)` | Fixed-ζ RU objective |

The constant distinguishes terminal returns from the donor's zero/nonterminal
flag. Costs enter L once. MSE needs no ζ calibration; ES calibrates ζ on separate
initial-policy paths and freezes it. This is not joint threshold/policy ES
optimization. Evaluation reports raw MSE, RMSE and ES separately.

`--market heston` and `--market gbm` use common presets and default to MSE.
`--config` loads the supplied financial components and follows their risk
objective unless `--objective` overrides it. These two input options are
mutually exclusive.

Search needs finite holding bounds. Minimum-order sizes are unsupported.
Lot-aligned grids require every target to be reachable from the initial
inventory; grid-to-grid trades must remain legal. A maturity-date action is
supported without drawing another market move.

| Supplied configuration | Decisions | Days/year | Hedge |
|---|---:|---:|---|
| [Heston stock-only](../benchmarks/configs/szehr-heston-stock.json) | 60 | 365 | Stock and cash |
| [GBM stock-only](../benchmarks/configs/szehr-gbm-stock.json) | 30 | 365 | Stock and cash |

Both sell one ATM call, use zero fees, omit hedge options, and bound stock
holdings between -1 and 1. `--grid-points 21` matches the donor action quantities.
These settings do not imply exact reproduction: donor Heston uses Euler and
hardcoded initial cash 0.021; the supplied configuration uses QE-M and
model-priced capital. The original clipped reward and cost-state offset are
also replaced by the objective above.

## Run

From a checkout with the baseline dependency group installed, supply the pinned,
explicitly modified donor outside the repository and a new external output
directory that does not yet exist:

```bash
uv run --frozen --group baselines python -m benchmarks.qualify_source_alphazero \
  --donor /path/to/minimalHedger_AlphaZero --output /path/to/source-az-run \
  --market heston --objective mse
```

Defaults use one cycle, eight episodes, 25 simulations per decision and two
supervised epochs. This is an execution check, not a convergence benchmark.
Overly small budgets can fail the runner's fitting/changed-weight checks.

For the supplied stock-only Heston configuration:

```bash
uv run --frozen --group baselines python -m benchmarks.qualify_source_alphazero \
  --donor /path/to/minimalHedger_AlphaZero --output /path/to/source-az-stock-run \
  --config benchmarks/configs/szehr-heston-stock.json --grid-points 21 --objective mse
```

Outputs include configuration, objective, donor provenance, policy snapshots,
replay examples and evaluation tapes. Weight-loading checks do not establish
interruption-exact optimizer resumption. MSE comparisons need MSE-trained
baselines on the same financial task.

## Accounting parity

The separate audit replays donor-generated paths and exact donor actions through
the common ledger. It harmonizes initial capital for that audit and records
both original premiums, raw losses and the original clipped reward.

```bash
uv run --frozen --group baselines --with scipy python -m benchmarks.check_source_alphazero_parity \
  --donor /path/to/minimalHedger_AlphaZero --output /path/to/existing-directory/parity.json
uv run --frozen pytest -q tests/test_source_alphazero.py tests/test_paper_benchmarks.py
```

The JSON file must be new and its parent directory must already exist. SciPy is
needed by the actual donor game modules. Exact fixed-input arithmetic does not
imply identical random streams, discretization, training or stochastic outcomes.
