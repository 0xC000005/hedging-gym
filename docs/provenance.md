# Provenance

This private repository is the maintained home for the shared hedging environment
and its method adapters. Thesis owns research decisions and literature. The
preceding Thesis-Experiments implementation remains historical evidence.

## Source snapshot

The initial import adapts `0xC000005/Thesis-Experiments` at source snapshot
`293dfc70b04109603e7f0d645baac0144c5b19d5`. This source commit was retained locally
and is not a published upstream reference. Source paths below are
relative to that repository, under
`directions/adaptive_option_control/heston_v1/` unless stated otherwise.

| This repository | Original source |
|---|---|
| `src/hedging_gym/{finance,gbm,bates}.py` | `finance.py`, `gbm.py`, `bates.py` |
| `src/hedging_gym/{gym_env,evaluation,benchmark}.py` | Same module names |
| `src/hedging_gym/{quickstart,validate}.py` | Same module names |
| `src/hedging_gym/__init__.py` | `public_api.py` |
| `README.md`, `docs/validation.md` | `PUBLIC_README.md`, `PUBLIC_VALIDATION.md` |
| `methods/` | Selected policy, classical-control, controller and training code from `policies.py`, `classical.py`, `common_eval.py`, `run.py` |

The migration uses a conventional `src` package and removes dependencies on
the old experiment package paths. It retains the financial equations and
separates learners from the market, ledger and evaluation implementation.
Historical checkpoints, generated outputs, full experiment history and trained
method rankings are not part of the initial import.

The historical numbers in [validation](validation.md) come from
`PUBLIC_VALIDATION.md`
and its detailed source reports: `environment-validation-2026-09-06.md`,
`heston-validation-2026-09-06.md`, `gbm-bates-validation-2026-09-06.md`, and
`bates-refinement-2026-09-06.md` in the same directory. Those reports identify
the retained external run artifacts. The standalone check commands establish
their own bounded local result and do not rerun those archived studies.

## Numerical and research lineage

The Heston characteristic function and Gauss–Legendre Fourier panels descend
from `directions/adaptive_option_control/finance_first_v1/benchmark.py`.
The QE/log-spot transition descends from
`directions/d1_nonconvex_hedging/gpc_transfer_v1/finance.py` and adapts the
Andersen construction used by
[PFHedge at commit `1fc08c7`](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/pfhedge/stochastic/heston.py).
It has no exact martingale correction and is not exact joint Heston simulation.
QuantLib provides independent references, not the environment's market paths or
cash-accounting implementation.

The risk-sensitive setting follows
[Bühler et al., Deep Hedging](https://arxiv.org/abs/1802.03042), with a longer-dated
call replacing their variance-swap hedge. Learned no-trade bands adapt the
PFHedge/[Imaki et al.](https://arxiv.org/abs/2103.01775) construction: a learned
center replaces the single-stock Black–Scholes anchor, with bounded widths per
asset. Ordinary pathwise training does not estimate moving fixed-ticket boundary
terms. Operational frictions are motivated
by [Maggiolo et al.](https://arxiv.org/abs/2510.01874), and observed-market
adaptation by [Schmid and Oeltz](https://arxiv.org/abs/2504.16436).
The book, presets, learner adapters and A → B → A protocol are disclosed
**adaptations**; they do not reproduce those papers' full experiments or imply
method superiority. Citations do not imply endorsement or grant a blanket
license to this repository or its dependencies.
