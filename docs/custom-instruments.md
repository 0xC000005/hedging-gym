# Custom instruments and settlement

An instrument defines its current mark and expiry payoff. It does not own the
simulator, change transaction costs or update the ledger. The same instrument
can be a hedge or the liability.

Built-ins are `EuropeanOption` and `VarianceSwap`. The latter follows Bühler's
unannualized variance-leg convention. A custom instrument implements the small
`Instrument` protocol: `maturity`, `kind`, `needs_integrated_variance`, numeric
`features`, and `mark(...)`.

For example, a claim paying squared stock under GBM at zero rates:

```python
from dataclasses import dataclass, field
import torch

@dataclass(frozen=True)
class SquaredStock:
    maturity: float
    kind: str = field(default="squared_stock", init=False)
    needs_integrated_variance = False

    @property
    def features(self):
        return {"maturity": self.maturity}

    def mark(self, spot, variance, remaining, market, *, integrated_variance=None):
        return spot.square() * torch.exp(variance * remaining)
```

`mark` receives the current batched spot and variance, remaining year fraction,
market parameters, and accumulated variance when requested. At zero remaining
time it must return the settlement payoff. Treat inputs as read-only, preserve
their batch shape/device, and use differentiable Torch operations when the
learner requires pathwise gradients. This example's formula is for **GBM**;
adding an instrument does not make its formula valid under other models.

Pass an instance directly into `PortfolioConfig`. To reload its dataclass fields
from JSON or a checkpoint, register its constructor in that process:

```python
from hedging_gym import PortfolioConfig, EuropeanOption, register_instrument

register_instrument("squared_stock", SquaredStock)
book = PortfolioConfig(
    liability=EuropeanOption(strike=1., maturity=30/365),
    hedges=(SquaredStock(maturity=60/365),),
)
```

No change to the engine's instrument list is needed. The current path-state
support is spot, variance and optionally integrated variance. Instruments
requiring another running state, early exercise or intermediate cash payments
need that capability implemented; a terminal pricing callback cannot invent it.

## Ending an episode

These choices are independent of the instrument and market:

- `TimeGrid(..., trade_at_maturity=True)` adds a final action at the maturity
  price, **without** another market move. `config.n_steps` counts market moves;
  `config.n_decisions` counts actions. Use the latter in rollout loops.
- `SettlementConfig(mode="liquidate")` closes positions and charges configured
  trading costs. Set `charge_liquidation_costs=False` to waive terminal fees.
- `SettlementConfig(mode="mark_to_market")` values remaining inventory without
  an extra trade, fee, turnover or ticket.
- `PortfolioConfig(initial_cash=..., initial_positions=...)` declares an
  existing endowment. By default positions are zero and cash is the model
  liability premium. Supplied inventory is already owned, not a free initial
  trade made by the policy.

`ExecutionConfig` independently defines proportional, per-unit, capped, fixed
and quadratic charges. A positive `commission_cap` caps the proportional plus
per-unit commission after its minimum; it does not cap fixed or quadratic
charges. Zero means uncapped. Setting both holding bounds to `None` permits
unbounded finite positions.

Price-proportional and quadratic charges use the absolute mark; a negative-value
custom claim does not turn a trading fee into a rebate.

For a new ending convention beyond these modes, extend the single
`settle_ledger` function and its independent NumPy counterpart. Keep that
accounting change out of instruments and learner callbacks.
