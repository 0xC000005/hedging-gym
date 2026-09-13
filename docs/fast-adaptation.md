# Fast-adaptation comparisons

These runners compare how a pretrained hedger is initialized and updated for a
new observed market. The supplied recipe uses the same Heston book, information,
execution rules and terminal ES95 across methods.

| Method | Target conditioning |
|---|---|
| Adaptive DH | Context appended to the policy input |
| [GEPS](geps-adaptation.md) | Context modulates internal weights |
| [SRSA-style retrieval](skill-retrieval.md) | Selected source context, followed by the same context updates |
| [Belief encoder + DH](belief-adaptation.md) | Context inferred from completed prior histories |

Target tasks start independently from the pretrained policy. This comparison
does not measure A→B→A forgetting. Market parameters remain observable.

## Generate banks and compare methods

Run commands from an installed repository checkout. Replace the placeholders
with external output directories. A small end-to-end execution check is:

```bash
uv run --frozen python -m benchmarks.compare_fast_adaptation banks \
  --output /path/to/adaptation-smoke --device cpu --smoke
uv run --frozen python -m benchmarks.compare_fast_adaptation compare \
  --output /path/to/adaptation-smoke --method adh --seed 7 --device cpu --smoke
```

For the full recipe, use a new directory and omit `--smoke` from both stages.
Run `compare` for each of `adh`, `geps` and `belief` and each seed 7, 17 and 29.
The `adh` arm also evaluates mean, nearest-market, learned top-one, top-five and
exhaustive context selection.

The runner defines eight source markets, three unseen target markets and one
seen reference. Full pretraining uses 6,000 updates; target curves use 0, 10, 50
and 200 updates. Exact configurations and disjoint seed families are in
[compare_fast_adaptation.py](../benchmarks/compare_fast_adaptation.py).
The recipe requires finite continuous holding bounds and does not support
minimum-order sizes or trade lots.

```bash
uv run --frozen python -m benchmarks.summarize_fast_adaptation /path/to/fast-adaptation
```

Summarize after completing the full method/seed matrix. Outputs include bank
metadata, source models, optimizer/RNG checkpoints and complete loss tapes.
Repeating an unchanged comparison resumes supported checkpoints and incomplete
curves. Use a fresh directory for a changed configuration or budget.

## Context-only versus full-network adaptation

This comparison starts both update modes from the same Adaptive-DH source
weights, calibration-selected context, threshold and minibatch stream.
It measures which parameters may change under a common optimizer recipe.

The source directory must contain `adh/seed-N/pretrained.pt` from the preceding
full comparison. Generate fresh target banks in a new directory:

```bash
uv run --frozen python -m benchmarks.compare_update_capacity banks \
  --output /path/to/capacity --device cpu
uv run --frozen python -m benchmarks.compare_update_capacity compare \
  --output /path/to/capacity --checkpoints /path/to/fast-adaptation --seed 7
```

Repeat `compare` for seeds 17 and 29, then run:

```bash
uv run --frozen python -m benchmarks.summarize_update_capacity /path/to/capacity
```

Source models are not retrained. Both modes pay for selecting their initial
context, and zero-update policies provide no-update controls.

## Adaptation-aware pretraining

[meta_pretraining.py](../src/hedging_gym/extensions/meta_pretraining.py) transfers
the first-order mechanism from
[MAML](https://proceedings.mlr.press/v70/finn17a.html), using
[author source pinned at a7f45f1](https://github.com/cbfinn/maml/tree/a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49).
The relevant `maml.py` operations stop inner gradients and evaluate post-update
query loss.

A copied policy adapts on support paths; its query gradients update the
initialization. Inner Adam, task contexts and exact empirical terminal ES replace
the source's supervised tasks and gradient-descent recipe. There are no second
derivatives or fast-weight interpolation. This implementation supports terminal
ES only. Source banks share all non-market configuration and are split into
disjoint support/query halves.

The `ordinary` control uses zero inner updates with the same outer architecture,
query objective and added forward-episode budget. Both continue from the same
original Adaptive-DH checkpoints.

```bash
uv run --frozen python -m benchmarks.compare_adapt_aware banks \
  --output /path/to/adapt-aware --device cpu
uv run --frozen python -m benchmarks.compare_adapt_aware train \
  --output /path/to/adapt-aware --source /path/to/fast-adaptation --family meta --seed 7
uv run --frozen python -m benchmarks.compare_adapt_aware develop \
  --output /path/to/adapt-aware --source /path/to/fast-adaptation \
  --development /path/to/capacity --family meta --seed 7
```

Train `meta` and `ordinary` for seeds 7, 17 and 29. Run `develop` for both and
for `original`, which reuses the source checkpoints. Select learning rates
using development markets only, then evaluate each family/seed:

```bash
uv run --frozen python -m benchmarks.compare_adapt_aware select \
  --output /path/to/adapt-aware --development /path/to/capacity
uv run --frozen python -m benchmarks.compare_adapt_aware test \
  --output /path/to/adapt-aware --source /path/to/fast-adaptation --family meta --seed 7
uv run --frozen python -m benchmarks.summarize_adapt_aware \
  --output /path/to/adapt-aware --development /path/to/capacity
```

Run the summary after completing all families and seeds. Final banks are
separate from development and source support/query banks. The optional
`diagnose` stage examines the inner-update recipe on development tasks.

## Interpretation

Compare zero-update risk and subsequent improvement. A better initialization
alone does not show a more effective update procedure. Charge source training,
all bank generation, history encoding, retrieval/rescoring and adaptation;
unique consumed paths and repeated rollout work are different quantities.
Concurrent wall times do not establish isolated algorithm speed.

The summaries reconstruct ES from saved losses and perform accounting checks.
Small seed sets and conditional path intervals do not establish a definitive
method ranking. The optional `benchmarks.attribute_pretraining` runner compares
RU versus empirical-ES continuation and batch sizes; its `--help` describes the
additional source/development artifact requirements.
