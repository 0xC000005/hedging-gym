# Validation

Prices, cash accounting, simulation precision and hedging risk are separate
questions. The commands below check selected implementations and interfaces.
Passing them does not establish market realism or a method ranking.

## Run the checks

From the repository root:

```bash
uv sync --locked
uv run --frozen python -m hedging_gym.validate --device cpu
uv run --frozen pytest
```

The first command installs the locked environment, including QuantLib and the
test dependencies. The second runs the bounded package validator. The third
runs the checkout test suite. Run `--device cuda` in place of `--device cpu`
when CUDA is available; the scalar Gymnasium checker still runs on CPU.

## What the checks establish

| Area | Check |
|---|---|
| Option prices | Float64 GBM, Heston and Bates prices against QuantLib in ordinary, short-expiry and stressed states; calls/puts on all supported year clocks |
| Greeks | Selected derivatives against bumped reference prices and difficult jump states |
| Cash accounting | Torch trade tapes reconstructed with a separate NumPy ledger, including fees and liquidation |
| Execution | Holding limits, minimum orders, lot increments, HOLD and terminal closeout |
| Configuration and interfaces | Independent components, variable portfolios, causal observations, scalar/vector/tensor behavior and gradients |
| Simulation refinement | Repeatability and unchanged decision dates with finer internal substeps |
| Method adapters | Small training/evaluation runs, shared-ledger consistency and policy schema checks |

The package validator prints its workload, seeds, device, tolerances and failures.
Its price probes use `atol=1e-8, rtol=1e-7`; its complete ledger replay uses
`atol=2e-6`. Test outcomes apply to the code and configuration that were run.
Available CUDA cases add device checks; skipped cases provide no CUDA evidence.
Gymnasium warnings about physical holding units and unbounded market
observations reflect the documented interface conventions.

## Numerical scope

GBM uses exact conditional lognormal stock steps at constant variance. Heston
defaults to QE-M (`scheme="qe_m"`), with the conditional stock martingale
correction from [QuantLib 1.43's Heston process](https://github.com/lballabio/QuantLib/blob/v1.43/ql/processes/hestonprocess.cpp).
Plain QE remains available as `scheme="qe"`. Bates adds compensated
compound-Poisson lognormal jumps to the selected Heston diffusion. Refining
internal substeps changes numerical resolution while preserving the decision
grid. Both schemes are approximate; the martingale correction does not make
Heston or Bates exact joint-path simulation.

Run the scheme-specific checks with:

```bash
uv run --frozen pytest -q tests/test_heston_schemes.py
```

QuantLib parity must select the same scheme: `QuadraticExponential` for QE or
`QuadraticExponentialMartingale` for QE-M. Agreement on fixed-input transitions
does not establish path-distribution convergence or policy tail-risk precision.

For option pricing, high volatility-of-variance requires a longer Fourier tail
than a cutoff based only on average variance. The tensor pricer uses the
asymptotic decay scale from [QuantLib 1.43's Heston engine](https://github.com/lballabio/QuantLib/blob/v1.43/ql/pricingengines/vanilla/analytichestonengine.cpp)
to extend its integration range. Price checks cover the source-matched Heston
defaults at short maturities and near-zero variance; an ATM delta is checked
separately. These checks do not establish path-distribution convergence, which
requires separate simulation-refinement checks before comparing training results.

**Bates tail-risk precision remains incomplete.** The automated suite does not
establish distributional convergence of Heston/Bates paths or policy tail losses.
Price agreement and cash reconciliation answer different questions from those
larger statistical experiments.

The supported financial domain has zero rates and dividends, European options,
observed market parameters and synchronized batches. Parameter changes in the
A → B → A sequence occur between episodes. Hidden regimes, within-episode
recalibration and empirical-market performance are outside the demonstrated
scope.

For a research comparison, declare the configuration, legal actions, initial
capital, risk level, seeds and compute budget before training. Evaluate frozen
policies on fresh paths, use multiple training seeds, report pooled terminal
cost-inclusive ES, and charge bank preparation and training separately from
decision-time evaluation. Requalify prices, cash accounting, simulation and
policy risk when moving beyond the checked domain.
