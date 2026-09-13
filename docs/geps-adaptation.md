# GEPS conditioning for Deep Hedging

[GEPSPolicy](../src/hedging_gym/baselines/geps.py) uses one task context to
modulate every layer of a shared hedger. Source training learns the shared
weights and source contexts; target adaptation freezes those quantities and
fits a new context. It uses the same observations, target holdings, cash ledger
and terminal-risk calculation as Adaptive Deep Hedging.

## Sources and implementation

The layer follows equations (4)–(5) of
[GEPS, NeurIPS 2024](https://arxiv.org/abs/2410.23889) and the
[author implementation](https://github.com/itsakk/geps/tree/e9a865218ecffacb7007ac7d719f3741afcf8c02),
pinned at `e9a865218ecffacb7007ac7d719f3741afcf8c02`. The relevant definitions
are `GEPSLinear` in `geps/model/layers.py` and `MLP` in
`geps/model/networks.py`.

For row-vector inputs, the layer computes:

```text
x @ W + ((x @ A) * c) @ B + b + c @ bias_context
```

This includes the context-dependent bias and is equivalent to forming
`W + A @ diag(c) @ B` explicitly. The same context is used in the output layer;
the adaptation factor is fixed at one.

The adapter independently expresses the published equations and does not
vendor the author's source. The pinned upstream repository has no tracked
license file. A separate source checkout is needed only for the optional
author-layer comparison.

## Supported use and deviations

Use `GEPSPolicy` through `train_multitask(..., policy_class=GEPSPolicy)` in
`hedging_gym.baselines.adaptive_deep_hedging`. It inherits finite holding bounds
and the continuous-action restriction: minimum-order sizes and trade lots are
unsupported. Source tasks must share the book, calendar, execution rules and
risk configuration; only market parameters may differ.

The supplied comparison uses terminal ES and a four-dimensional context.
Joint source training has no inner meta-gradient loop. Target adaptation fits
the context and ES threshold with the shared network frozen.

Compared with the PDE source, this implementation uses bounded hedge holdings,
terminal financial risk, and the common two-hidden-layer tanh policy. Source
contexts start at zero. Layer initialization follows `GEPSLinear.reset_parameters`
rather than dataset-specific overrides, but does not reproduce the author's
random-number consumption. Low-rank layers add shared parameters, so this is
not a parameter-matched comparison with Adaptive DH.

## Run

From an installed checkout, replace the output placeholder with a directory
outside the repository:

```bash
uv run --frozen python -m benchmarks.compare_fast_adaptation banks \
  --output /path/to/fast-adaptation --device cpu
uv run --frozen python -m benchmarks.compare_fast_adaptation compare \
  --output /path/to/fast-adaptation --method geps --seed 7 --device cpu
```

The full recipe performs substantial pretraining. Add `--smoke` to both stages,
using a separate directory, for a small execution check. See
[fast adaptation](fast-adaptation.md) for controls, additional seeds and summaries.

Focused equation, gradient and adaptation checks:

```bash
uv run --frozen pytest -q tests/test_geps.py tests/test_adaptation_methods.py
GEPS_DONOR_ROOT=/path/to/pinned-geps uv run --frozen pytest -q tests/test_geps.py
```

Without `GEPS_DONOR_ROOT`, the author-layer comparison is skipped. These checks
establish implementation behavior, not PDE reproduction or improved hedging risk.
