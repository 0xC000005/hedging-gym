# Benchmark and financial contract

The benchmark is a small, explicit experiment for comparing hedging methods
under a common terminal-loss objective. Its parameters and execution charges
are synthetic assumptions. They are not a market calibration or a claim that
one method is best.

## Default experiment

`benchmark_config()` selects the following settings. Passing a component
replaces that component; bare market configurations contain no portfolio or
trading calendar. `name=...` supplies execution defaults only when `execution`
is omitted. An explicit `ExecutionConfig` replaces the whole component,
including zero fees; use `operational_config` to deliberately overlay it later.

| Component | Default |
|---|---|
| Market | Heston, with initial spot 1 and variance 0.04 |
| Heston parameters | Mean reversion 3, long-run variance 0.04, volatility of variance 0.3, correlation −0.5 |
| Time grid | 30 decisions, one step per 1/252 year |
| Liability | One ATM European call owed, settling after the last interval |
| Hedge instruments | Stock and one ATM European call maturing at twice the episode horizon |
| Initial positions and cash | No hedge holdings; cash equals the signed liability premium |
| Holding bounds | Stock [−1, 2]; each hedge option [−1, 1] |
| Proportional fees | 0.0005 of stock notional traded; 0.01 of option premium traded |
| Other execution charges | Zero |
| Risk | Expected shortfall at confidence 0.95 |

Choosing `model="gbm"` uses constant variance 0.04 and physical drift zero.
Choosing `model="bates"` adds independent jumps to the Heston diffusion, with
intensity 1 per year, normal log-jump mean −0.1 and standard deviation 0.2.
An explicit market object can change these parameters without changing the
other components.

For a custom time grid, the default liability matures at `time_grid.horizon`
and the default hedge at twice that horizon. Custom portfolios may use European
calls or puts, any number of hedge options including zero, and a signed
liability quantity. Positive liability quantity is owed; negative quantity is
owned. Hedge positions count options with unit underlying multiplier. Cash is
accounted separately and is not an action-space instrument.

## Cash, trades and loss

Let decisions occur at $t_i=i\Delta t$, $i=0,\ldots,N-1$, with settlement at
$T=t_N$. Let $M_i$ be the vector of stock and hedge-option mid-prices, and $h_i$
the target holdings after decision $i$. Starting from $h_{-1}=0$, the trade is

```math
\Delta h_i=h_i-h_{i-1}.
```

Let $Q$ be the signed liability quantity and $V_0$ its unit initial model price.
Initial cash is $B_{-1}=QV_0$. At zero interest and dividend rates, the
self-financing cash update is

```math
B_i=B_{i-1}-\Delta h_i^{\mathsf T}M_i-C_i(\Delta h_i).
```

At settlement, all remaining hedge holdings are closed at their current marks
and the signed liability payoff $Q\Phi(S_T)$ is paid:

```math
\Pi_T=B_{N-1}+h_{N-1}^{\mathsf T}M_N
      -C_N(-h_{N-1})-Q\Phi(S_T),
\qquad L=-\Pi_T.
```

Here $\Phi(S_T)=(S_T-K)^+$ for a call and $(K-S_T)^+$ for a put. A hedge option
expiring at $T$ is closed at its payoff; a later-expiring hedge is sold at its
model mark. Both closeouts incur the configured transaction costs. Intermediate
hedge expiries are unsupported, and liability maturity must equal $T$.

For instrument $j$, write $p_j$ for the proportional rate, $a_j$ for the
quadratic rate, $f_j$ for a fixed ticket and $m_j$ for a minimum commission.
The charge for trade $x_j$ at mid-price $M_{i,j}$ is

```math
c_{i,j}(x_j)=\mathbf{1}_{\{x_j\ne0\}}
 \left[\max\left(p_j|x_j|M_{i,j},m_j\right)+f_j\right]
 +a_jx_j^2M_{i,j},
\qquad C_i(x)=\sum_j c_{i,j}(x_j).
```

HOLD therefore costs zero. A minimum commission replaces a smaller proportional
charge; it is not added to it. Position limits constrain $h_i$. A nonzero trade
must meet its minimum order size and, where configured, be an integer multiple
of its lot size. Forced final liquidation waives the minimum order size but
retains lot rules and every fee. The environment rejects illegal targets rather
than silently projecting them.

## Risk objective

The configured risk measure is upper-tail expected shortfall of terminal loss:

```math
\mathrm{ES}_{\alpha}(L)
=\frac{1}{1-\alpha}\int_{\alpha}^{1}\mathrm{VaR}_u(L)\,du
=\min_{\zeta\in\mathbb{R}}
 \left\{\zeta+\frac{\mathbb{E}[(L-\zeta)^+]}{1-\alpha}\right\}.
```

This optimized-certainty-equivalent representation is also used in
[Deep Hedging, Section 3.2](https://arxiv.org/html/1802.03042v1#S3.SS2).

`RiskConfig(alpha=...)` chooses $0<\alpha<1$. The training adapters learn one
global threshold $\zeta$ with the policy, using the Rockafellar–Uryasev loss on
complete episodes. The evaluator pools all complete losses and computes
empirical ES with fractional weight at the tail boundary. Its primary metric
is `expected_shortfall`, accompanied by `risk_alpha`; ES95 and ES99 are additional
diagnostics. Averaging minibatch ES gives a different statistic.

By default, Gym rewards are zero before settlement and equal $\Pi_T$ at
settlement. Changing `RiskConfig` alone does not change that reward: an ordinary
expected-return learner still optimizes mean P&L.

Passing `risk_threshold=zeta` to either Gym wrapper instead returns
`-config.risk.loss(terminal_loss, zeta)` at settlement, using `config.risk.alpha`.
The learner fits this global threshold on training data; the environment does
not learn it. The tensor training adapters apply the same risk loss directly
to complete episode batches.

## Market models and information

GBM uses exact conditional lognormal stock steps with constant variance $v_0$.
Its configurable physical drift affects simulated paths; risk-neutral option
pricing uses the required zero-rate, zero-dividend convention. Heston evolves

```math
\frac{dS_t}{S_t}=\sqrt{v_t}\,dW_t^S,\qquad
dv_t=\kappa(\theta-v_t)\,dt+\sigma\sqrt{v_t}\,dW_t^v,
\qquad d\langle W^S,W^v\rangle_t=\rho\,dt.
```

Bates adds compensated compound-Poisson lognormal stock jumps. The physical and
pricing jump parameters coincide here. Heston/Bates stock and variance paths
use an approximate quadratic-exponential/log-spot diffusion step. Prices use
their characteristic functions; GBM prices use Black–Scholes, and put prices
use zero-rate put–call parity. See the [numerical references](related-work.md)
and [validation limits](validation.md).

Observations contain current time, market state, cash, positions, marks, market
parameters, contract terms and execution rules. They do not contain future
path values. Their schema depends on the chosen market and portfolio. All
controllers must derive their dimensions from that schema. Internal simulation
substeps refine the market integrator without adding trading decisions.

`hedging_gym.finance.observation_fields(config)` lists columns in their exact
order. Instrument blocks follow stock, then `portfolio.hedges`.

| Observation fields | Values supplied to the controller |
|---|---|
| `time_fraction` | Decision index divided by the number of steps |
| `spot`, `cash`, instrument mids | Divided by `market.spot0` |
| `variance` | Divided by `max(market.v0, 1e-4)` |
| Instrument positions | Stock/option quantities, not normalized |
| Execution rules and market parameters | Their configured values |
| Contract strikes, maturities, quantity, `dt`, `spot0`, `v0` | Their configured values; maturities and `dt` are in years |
| Liability and hedge kinds | +1 for calls, -1 for puts |

`decode_market_observation(observed, config)` recovers spot and variance in model
units. Risk confidence and the learned ES threshold are not observation fields.

## Execution presets and regime changes

`operational_config(base, name=...)` changes only execution rules. It preserves
unspecified fields, so overlays can be composed. Explicit execution overrides
take precedence over the named preset. `basic` is an empty overlay.

| Preset | Field set |
|---|---|
| `operational_fixed` | Fixed ticket 0.0001 for every instrument |
| `operational_minimum_fee` | Minimum commission 0.0001 for every instrument |
| `operational_minimum_trade` | Minimum nonzero increment 0.01 stock, 0.1 per option |
| `operational_lots` | Trade-increment grid 0.001 stock, 0.1 per option |

Amounts are expressed in the normalized book's units. For illustration, with
initial stock price \$100 and 1,000 underlying units, a stock holding of 0.001
represents one share, an option holding of 0.1 represents one 100-share contract,
and a money charge of 0.0001 represents \$10. This interpretation does not make
the preset charges broker-calibrated fees.

```python
from hedging_gym import benchmark_config, operational_config, adaptation_configs

base = benchmark_config(model="heston")
base = operational_config(base, name="operational_fixed")
stages = adaptation_configs(base)
```

The default stages are A → B → A. B changes initial variance from 0.04 to 0.09
and, for Heston/Bates, long-run variance to 0.09. Explicit `market_changes` may
change stochastic parameters within the selected model family. Portfolio,
calendar, execution and risk stay fixed. Regimes change observably between
independent episodes.

`evaluate_adaptation` evaluates before and after the requested update callbacks
at each stage. The same caller-owned learner persists across stages. Training,
pre-update evaluation and post-update evaluation use separately seeded banks.
Return-A before updating measures forgetting relative to original-A after
updating; return-A after updating measures recovery. Evaluation must leave the
learner frozen. This is a protocol for testing adaptation, not evidence that a
particular learner adapts successfully.

## Why these choices

[Deep Hedging](https://arxiv.org/abs/1802.03042v1) motivates learning constrained
hedging strategies under transaction costs and terminal risk objectives. The
paper's Heston example uses stock and a variance swap; this benchmark instead
uses a European option. That extra hedge gives a second sensitivity to trade against
stock exposure. Its tenor, strikes, holding bounds and fee levels are explicit
benchmark choices; they are not attributed to a paper as calibrated constants.

[Maggiolo et al.](https://arxiv.org/abs/2510.01874v2) motivate studying non-convex
execution settings. The separate fee/order presets allow those effects to be
examined while holding market dynamics fixed. [Schmid and Oeltz](https://arxiv.org/abs/2504.16436v1)
motivate studying changes in market parameters. The observed A → B → A sequence
here isolates that question while keeping the portfolio and operational rules
fixed. The [related-work guide](related-work.md) identifies these connections
without claiming reproduction of the papers' algorithms or results.
