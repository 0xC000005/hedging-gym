# Hedging Gym

![Status: research alpha](https://img.shields.io/badge/status-research%20alpha-orange)
![Python 3.12–3.13](https://img.shields.io/badge/python-3.12%E2%80%933.13-blue)
![Gymnasium 1.2.3](https://img.shields.io/badge/Gymnasium-1.2.3-blue)

A small, batched environment for option-hedging research. GBM, Heston and Bates
share one cash ledger, legal-action checks and terminal-risk evaluator.
Gymnasium and PyTorch interfaces run on CPU, with CUDA support for tensor work.
Classical controls, Deep Hedging and learned no-trade bands provide initial
method adapters outside the environment package.

**ADAPTATION / research alpha.** Heston has the strongest historical numerical
qualification; Bates tail-risk precision remains incomplete. This repository
provides runnable implementation checks, not a completed algorithm ranking or
a reproduction of the cited papers. See [validation](docs/validation.md) and
[provenance](docs/provenance.md).

## Install, check and try

With [uv installed](https://docs.astral.sh/uv/getting-started/installation/) and
access to this private repository:

```bash
git clone https://github.com/0xC000005/hedging-gym.git
cd hedging-gym
uv sync --locked
uv run --frozen python -m hedging_gym.validate --device cpu
uv run --frozen python -m hedging_gym.quickstart --device cpu --paths 64
```

The locked install creates `.venv` and installs the package, tests and
**QuantLib 1.43 automatically**. No separate QuantLib installation is needed on
a supported platform. See [uv's project workflow](https://docs.astral.sh/uv/guides/projects/).
The alpha is not published to a package index. Python 3.12–3.13 is supported by
the package metadata; Linux/Python 3.12 is the qualification platform.

For the one-command financial validation workflow, including dependency
installation, independent references and focused tests:

```bash
bash scripts/validate.sh
```

The quickstart runs four complete batches: Basic, then observed A → B → A with
a constant operational overlay. Trades are scripted; 64 paths demonstrate the
API and do not estimate ES95 precisely. `--device cuda` is optional and requires
a compatible device and PyTorch installation.

## A complete batch

```python
import torch
from hedging_gym import HedgingVectorEnv, benchmark_config, empirical_es

config = benchmark_config(model="heston")
env = HedgingVectorEnv(64, config, device="cpu")
observation, info = env.reset_tensor(seed=42)

# Target holdings [stock, hedge call]; deliberately unhedged in this example.
targets = torch.zeros((64, 2), device=observation.device)
for _ in range(config.n_steps):
    observation, reward, terminated, truncated, info = env.step_tensor(targets)

print("Terminal loss ES95:", empirical_es(info["terminal_loss"], 0.95))
env.close()
```

`HedgingEnv` exposes the scalar Gymnasium API. `HedgingVectorEnv` offers NumPy
`reset`/`step` and PyTorch `reset_tensor`/`step_tensor` over the same state. Batch
members finish and reset together; autoreset is disabled. A seed reproduces a
fixed batch size, device and integration resolution. Observations contain
current market parameters and execution rules, with no future path values;
observation dimensions depend on the selected market.

Actions are target holdings in physical units. The action `Box` gives outer
bounds; minimum orders and lots also depend on current holdings. Use
`action_mask` or `action_mask_tensor` to check candidate actions. Invalid trades
raise an error. The tensor ledger preserves continuous gradients; hard ticket
activation and discrete lots require a suitable learning method. Gym's checker
warns about physical-unit bounds and unbounded market observations; these are
declared conventions.

## Financial contract

The default benchmark sells one ATM call expiring after 30 daily decisions and
hedges with stock, one ATM call expiring after 60 trading days, and cash.
Initial spot is one; rates and dividends are zero. The initial premium, every
trade, all fees, final hedge liquidation and liability settlement enter terminal
P&L. Holdings are bounded by `[-1, 2]` for stock and `[-1, 1]` for the hedge call.
Basic proportional fees are `0.0005` of stock notional and `0.01` of hedge-call
premium traded. These are synthetic assumptions, not calibrated broker fees.

Market dynamics, execution settings and learner updates are independent axes:

| Selection | Meaning |
|---|---|
| `benchmark_config(model="heston")` | Default market; also `"gbm"` and `"bates"` |
| `operational_config(base, "operational_fixed")` | Add `0.0001` per instrument traded |
| `"operational_minimum_fee"` | Commission floor `0.0001` per nonzero instrument trade |
| `"operational_minimum_trade"` | Minimum nonzero increments `(0.01, 0.1)` |
| `"operational_lots"` | Trade-increment grid `(0.001, 0.1)` |
| `adaptation_configs(base)` | Independent episodes in observed regimes A → B → A |

`operational_config` preserves unspecified fields, allowing overlays to compose;
`"basic"` changes nothing. Default regime B raises initial variance from `0.04`
to `0.09`, and long-run variance likewise for Heston/Bates. Execution stays fixed
through A → B → A. Regimes change between episodes and are observed.
`simulation_substeps` refines integration without adding trading decisions.

For a nonzero trade `dq` at price `mid`, each instrument pays
`max(proportional * abs(dq) * mid, minimum_commission)
+ quadratic * dq**2 * mid + fixed_ticket`; HOLD costs zero. Forced liquidation
waives the discretionary minimum order size, retaining lot rules and all fees.

Reward is zero until settlement, then terminal P&L. Maximizing mean reward does
not optimize ES95. `evaluate_controller` accepts a controller
`(observation, ledger, time_index, config) -> target_holdings` and a market bank,
returning metrics and actual trade tapes. ES95 pools complete-path losses with
fractional weight at the empirical tail boundary; do not average minibatch ES.
Bank preparation and training are separate from reported evaluation time.

`evaluate_adaptation` retains the caller's controller and update callback
through A → B → A, using separately seeded training and evaluation banks. It
evaluates before and after updates. Return-A before updates measures forgetting
relative to original-A after updates; return-A after updates measures recovery.
The caller supplies the learner, which must remain frozen during evaluation.

## Methods and repository layout

```bash
uv run --frozen python -m experiments.baselines
```

The default CPU development run uses 128 training paths, 128 held-out paths,
30 trading dates and eight updates per learned method. It trains Deep Hedging
and learned no-trade bands, then evaluates them alongside classical controls on
the same separate held-out bank. It uses the basic execution profile. See the
[method notes](methods/README.md) for attribution and CLI options. Smooth DH/band
adapters do not yet handle minimum-order and lot constraints. Small-run metrics
demonstrate the workflow; they do not establish method superiority. Broader RL, continual
adaptation and AlphaZero adapters remain migration work.

- `src/hedging_gym/`: shared market simulation, pricing, ledger, Gym/tensor APIs
  and terminal-risk evaluation; installed as `hedging_gym`.
- `methods/`: attributed method adapters, kept outside the finance core.
- `experiments/`: comparisons run from a repository checkout.
- `tests/` and `scripts/validate.sh`: executable financial and API checks.
- `docs/`: validation scope and source provenance.

Keep raw results and checkpoints outside Git. Research decisions and literature
remain in Thesis; this repository owns the common environment and adapters.
