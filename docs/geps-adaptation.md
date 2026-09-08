# GEPS conditioning on the common hedger

This is a finance adaptation of the GEPS layer mechanism, not a reproduction of
its PDE experiments or evidence of improved financial performance.

## Pinned authority

- Kassaï Koupaï et al., NeurIPS 2024, equations (4)--(5) and section 4.2:
  <https://arxiv.org/abs/2410.23889>.
- Author implementation: <https://github.com/itsakk/geps> at commit
  `e9a865218ecffacb7007ac7d719f3741afcf8c02`.
- Exact reference: `geps/model/layers.py`, `GEPSLinear`, lines 156--197;
  shared context across layers: `geps/model/networks.py`, `MLP`, lines 7--32.
- Checkout retained outside Git at
  `/home/max/Documents/hedging-gym-runs/r2-fast-adaptation-2026-09-08-DChV47/donors/geps`.
  No donor package is installed. The pinned repository has no tracked license
  file; this adapter independently expresses the published equations and does
  not vendor the author's source.

## Mechanism and integration

For row-vector inputs, each layer computes

`x @ W + ((x @ A) * c) @ B + b + c @ bias_context`.

This is exactly `x @ (W + A @ diag(c) @ B) + b + c @ bias_context`, including
the context-dependent bias from equation (5). One context `c` is used in every
layer, including the output. The reassociation avoids per-path dense adapted
weight matrices. The fixed adaptation factor is one.

`GEPSPolicy` subclasses `TaskEmbeddedPolicy` and accepts the same constructor
arguments. Its `shared` module splits the inherited concatenated input into
financial features and context: the context modulates weights, rather than
being an additional financial input. All observed market parameters remain in
the financial features. Existing action bounds and cash accounting are unchanged.

Use the common multitask trainer's policy-class hook to construct `GEPSPolicy`.
The existing optimizer groups fit `shared` and `source_embeddings` jointly with
one ES threshold per source task. This is first-order joint training; it is
**not** CAVIA-style differentiation through an inner adaptation loop. After
pretraining, inherited `prepare_adaptation()` freezes every shared parameter
and source context, sets the new context to the source mean, and adapts only
that context plus the common ES threshold through `AdaptationUpdater`.

The adaptation context has the same default four scalars as Adaptive DH. GEPS
adds shared low-rank parameters, so total model size is not parameter-matched;
record it and charge actual training/adaptation time. A faster or better result
must be measured, not inferred from low rank.

## Deliberate changes from the native PDE setup

- The output is bounded hedge holdings, and the training loss is the common
  cost-inclusive terminal ES objective, not PDE trajectory error.
- Common two-hidden-layer tanh architecture is retained instead of the author's
  four-layer contextual-Swish MLP and PDE integration stack.
- Source contexts start at zero as in the author's `Derivative`, rather than
  the comparator's `0.1 * randn`. Layer initialization uses the distributions
  in `GEPSLinear.reset_parameters`, not dataset-specific `init_weights` overrides.
  Its row-vector weight orientation is retained, including its initialization
  fan convention. Exact source random-number consumption is not reproduced.
- Source task schedules, path banks, optimizer settings, mean-context restart,
  ES-threshold fitting and evaluation separation belong to the common harness.
  This lane does not supply a separate training or evaluation protocol.

## Qualification

`tests/test_geps.py` checks the explicit equation and all parameter/input/context
gradients in float64, then compares the adapter to the actual pinned author
layer. It also checks joint source differentiation, one context across all
layers, the inherited context-only freeze/update contract, and a tiny common
ledger rollout. The author test needs `GEPS_DONOR_ROOT`; without it only that
source-dependent test is skipped.

These checks qualify equations and integration, not native PDE performance or
financial superiority. No full training belongs to this implementation gate.

On 2026-09-08, the source-dependent tests and existing adaptation tests passed:
`8 passed`, using the existing hedging-gym virtualenv, CPU, two OpenMP/MKL
threads, the worktree's `src` and root on `PYTHONPATH`, and the checkout above
as `GEPS_DONOR_ROOT`. Command: `python -m pytest -q tests/test_geps.py
tests/test_adaptation_methods.py`. The ledger test uses only two tiny updates;
no scientific training or GPU job was run for this gate.
