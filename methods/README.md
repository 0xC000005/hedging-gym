# Initial baseline adapters

These checkout-only modules begin the algorithm migration. The installable
`hedging_gym` wheel contains the environment and evaluator; it does not contain
learners. All methods use the same current observation, initial capital,
execution constraints, cash ledger and complete terminal-loss evaluator.

The source is private `Thesis-Experiments` revision
`293dfc70b04109603e7f0d645baac0144c5b19d5`, under
`directions/adaptive_option_control/heston_v1/`:

| Adapter | Retained source and adaptation |
| --- | --- |
| `policies.py` | `DirectDHPolicy`, `NoTransactionBandPolicy` and shared helpers from source `policies.py`; network and action equations retained. |
| `classical.py` | Source spot delta, gamma and variance Greeks, bounded stock/call sensitivity matching and stock band; only core imports and obsolete compatibility aliases changed. |
| `controllers.py` | Classical controller and deterministic policy adapter from source `common_eval.py`, exposing no future bank state. |
| `training.py` | Source `run.py` causal rollout, global Rockafellar–Uryasev ES95 threshold, Adam parameter groups, independent minibatch generator and gradient clipping. The small rollout calls `TensorHedgingEnv`; there is no second cash ledger. |

The direct bounded-target policy is a finance adaptation in the
[Deep Hedging](https://arxiv.org/abs/1802.03042) tradition. The learned band
retains the established clamp-to-band construction attributed in the donor to
[PFHedge's pinned NoTransactionBandNet example](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/README.md)
and [Imaki et al.](https://arxiv.org/abs/2103.01775). It learns a center and two
bounded widths per instrument in the shared market/portfolio state. This is
an **ADAPTATION**, not a literal paper reproduction or a new architecture.
Classical controls match current model-price derivatives under shared caps;
they do not minimize ES. Variance sensitivity is `d/dv`, not `d/dsqrt(v)`.

Run the development example from the repository root:

```bash
uv run --frozen python -m experiments.baselines
uv run --frozen python -m experiments.baselines --device cuda
```

Defaults use `benchmark_config("basic")`, keeping the common stock/60-day-call
book, complete execution-feature observation and 30 daily decisions, and train
DH and NTB for 8 updates each (batch 32; hidden layers 32, 32) on 128 paths,
then evaluate both plus delta, a predeclared 0.05 stock-quantity delta band,
delta-gamma and delta-variance on one shared, separate 128-path bank.
Policy seed is 7; training bank seed 1101; held-out bank seed 2201. One CPU
thread is the default. The selected device also generates the banks; equal
seeds across CPU/CUDA are not a promise of identical numerical paths.

`--output-dir /path/outside/git` optionally saves compact metadata, final policy
weights, the held-out market bank and each controller's holdings/loss tape for
independent cash reconstruction. Models are frozen before the evaluation bank
is generated; no checkpoint or band is selected on those results. Progress
prints configuration, work, seeds, device, timings and estimates to completion.

This tiny run checks integration only. Its training budget and tail sample are
too small for a performance claim. Timing reports bank preparation, learner
setup/training, and whole evaluator time separately; no speedup claim is made.
A formal comparison still needs declared budgets, strong developed baselines,
fresh paths, multiple training seeds, and independent financial qualification.

Continuous DH/NTB do not yet have an action parameterization for minimum-order
sizes or contract lots. Training rejects those configurations. All controller
targets reach the core unchanged and illegal trades raise; no rounding or
straight-through estimator is inserted. Fixed-ticket fees also create missing
moving-boundary terms in ordinary pathwise training, especially for learned
hold bands. These controls are not claimed to be qualified across every
operational overlay. Hybrid HPO, native external stacks, impulse policies,
adaptation schedules and future architectures are outside this initial import.
