# Validation

**ADAPTATION / numerical qualification.** Prices, cash accounting, simulation
precision and policy risk are separate checks. Agreement with QuantLib is
independent numerical evidence; it does not establish empirical-market realism
or an algorithm ranking.

## Reproduce the standalone checks

From a checkout with uv installed:

```bash
bash scripts/validate.sh
```

The script installs the locked environment, including QuantLib 1.43, then runs
the package validator and focused financial/API tests. Pass `cuda` as its final
argument to select GPU validation. The bounded installed-package validator and
the full repository test suite can also be run directly:

```bash
uv sync --locked
uv run --frozen python -m hedging_gym.validate --device cpu
uv run --frozen pytest
```

The package validator checks nine float64 call prices against QuantLib: ordinary,
short-expiry and higher-variance states for GBM, Heston and Bates. The tolerance
is `atol=1e-8, rtol=1e-7`. It replays eight complete Heston trade tapes with a
separate float64 NumPy ledger, checking terminal loss, costs, turnover, tickets,
holdings legality and forced liquidation (`atol=2e-6`). The probe composes fixed
tickets, minimum commissions, minimum trades and lots. It also runs Gymnasium's
scalar API checker. The runner prints seeds, device, work, tolerances and results;
failure raises an error.

The focused tests additionally cover 12 ordinary, short-expiry and rare Heston
price states against QuantLib, a delta against bumped QuantLib prices, and a
cash-ledger derivative against NumPy finite differences. They check shared
accounting across all three markets, observation causality, operational
feasibility, Gym/tensor gradients and the separation of market changes from
execution rules. The refinement test checks repeatability and unchanged trading
dates; it does not establish convergence or distribution accuracy.

GBM/Bates price and spot-Greek tests also retain the original four contract
states from two failed Bates jump paths. Their prices and gammas are compared
with tighter QuantLib integration and higher Fourier quadrature on CPU/CUDA,
so the previously fixed large-jump defect has an executable regression test.

`--device cuda` adds same-bank CPU/CUDA terminal-loss agreement when CUDA is
available. The scalar Gym check still runs on CPU. Financial action units and
unbounded observations cause documented Gym warnings. These small checks are
installation and implementation evidence, not a transition-law certificate or
the larger studies below.

## Standalone migration verification — 6 September 2026

- The locked checkout passed **37 tests**: 29 core/reference cases and eight
  initial method-adapter cases. Available CUDA cases ran, rather than skipped.
- Installed-package validation passed on CPU and CUDA; maximum error across
  its nine float64 QuantLib price probes was `2.44e-15`. The operational ledger
  and shared-bank CPU/CUDA replay passed their `2e-6` absolute tolerance.
- A wheel built from the source archive installed non-editably in a fresh
  Python 3.12 environment. Validation passed from outside the checkout under
  Python isolated mode; neither historical experiment imports nor checkout-only
  method modules were available. The wheel contains the financial core only.
- The 30-date development command ran all six controls on CPU and CUDA using
  128 training and 128 held-out paths, eight updates for each learned policy.
  All 12 saved method tapes independently reconciled with the NumPy ledger;
  maximum terminal-loss difference was `4.01e-7`. Those small, one-seed runs
  verify integration, not relative hedging quality or accelerator speedup.

The test script and demo reproduce these types of checks. Raw local migration
logs/checkpoints are kept outside Git; the research hub retains their location.

## Historical source evidence, recorded 6 September 2026

The following results are retained from the source snapshot identified in
[provenance](provenance.md). **They are reported historical evidence, not newly
rerun standalone results.** The original frozen policy checkpoints, large path
banks and full study runners are not included here; this checkout cannot fully
reconstruct those studies. Their conclusions apply to the specified source
configurations and frozen policies.

### Independent prices and cash replay

For each market, 512 complete paths were independently repriced at all 31 dates
using QuantLib 1.43: 47,616 observed states in total. Eleven fixed-trade tapes
were replayed with separate float64 NumPy cash calculations, including original
premiums, every fee, final liquidation and independent liability settlement.

| Market | Max float64 price error | Max saved float32 mark error | Max full-tape loss error |
|---|---:|---:|---:|
| GBM | 3.33e-16 | 6.68e-8 | 1.30e-6 |
| Heston | 2.26e-11 | 6.30e-8 | 1.30e-6 |
| Bates | 9.46e-10 | 7.67e-8 | 1.17e-6 |

All errors met the declared absolute tolerances `1e-8`, `2e-7` and `1e-5`,
respectively, with initial spot normalized to one. Tickets matched and holdings
were legal. A second cash equation reconstructed all 5,632 tape-path losses.
Fixed-trade replay checks calculations, not how a policy reacts to different
reference marks. Saved float32 states introduce input-rounding error, reported
separately from identical-state float64 pricing error.

Independent engines include QuantLib's pinned
[Heston engine](https://github.com/lballabio/QuantLib/blob/v1.43/ql/pricingengines/vanilla/analytichestonengine.cpp)
and [Bates process](https://github.com/lballabio/QuantLib/blob/v1.43/ql/processes/batesprocess.cpp).

### Dynamics and terminal risk

| Market | Completed source evidence | Unresolved precision and scope |
|---|---|---|
| Heston | 262,144 paths for each base/stressed configuration at 1, 4 and 16 internal substeps. Four frozen controls on 16,384 paths per source grid and independent QuantLib QE-martingale paths. All 12 marginal ES95 stability intervals fit the declared margin. | 25 of 36 moment/payoff checks met tight precision targets; 11 remained unresolved. Frozen-policy stability covered the base market only. Both simulators are discretizations. |
| GBM | Exact conditional lognormal steps at constant variance; Black–Scholes prices, Greeks, sampled payoffs and independent cash replay checked. | Exact transition sampling does not make finite-sample ES exact or remove discrete hedging risk. |
| Bates | Compensated compound-Poisson lognormal jumps, prices/Greeks and full cash replay checked. Refinement used 65,536 paths per source 1/4/16 grid and independent QuantLib FullTruncation16. | Only 2 of 6 ES95 intervals fit the declared ±5% margin; 4 were too wide. Only 4 of 24 distribution checks met tight precision targets; 20 remained unresolved. |

Heston's practical ES95 margin was ±`0.00062120657`, 5% of a previously frozen
reference control's ES95. Bootstrap intervals are marginal 95% intervals
conditional on frozen policies and action randomness, not simultaneous intervals
or uncertainty over training seeds. Approximate 99.9% Monte Carlo distribution
screening intervals containing their targets did not turn unresolved tight
precision checks into passes. These checks do not certify the full joint path
law or exact-Heston/Bates extreme tails.

A larger Bates run exposed invalid Greeks after two large downward jumps.
The corrected source uses 96 Fourier quadrature nodes per panel at positive
jump intensity, with the original failing states checked against tighter
reference prices, finite-difference Greeks and higher quadrature on CPU/CUDA.
Heston and zero-jump Bates retain 48 nodes. No failed path was removed and no
financial parameter or comparison threshold was relaxed.

## Limits for research use

Heston has direct historical frozen-policy risk-stability evidence. GBM has an
exact constant-variance transition. Bates prices and accounting have independent
checks, while its tail-risk precision remains incomplete. None is uniformly
certified over every parameter, payoff or tail event.

The default book has zero rates/dividends, observed parameters, synchronized
batches and fixed daily decisions. Hidden regimes, changing intraday parameters
and empirical-market performance are outside the tested scope. The observed
A → B → A runner is a data/execution contract, not a completed trained-adaptation
comparison. Initial method adapters and small development runs are adaptations,
not faithful paper reproductions or multi-seed performance evidence.
