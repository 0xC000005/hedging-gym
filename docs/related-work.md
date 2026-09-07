# Related work

These references explain the research ideas and numerical constructions used
here. The benchmark and method adapters make their own disclosed choices;
they do not reproduce the cited papers' complete experiments.

| Reference | Connection to this repository |
|---|---|
| Bühler, Gonon, Teichmann and Wood, [Deep Hedging](https://arxiv.org/abs/1802.03042v1) | Learning constrained hedging strategies under transaction costs and a terminal risk objective. The benchmark uses a tradable European option as an additional hedge. |
| Imaki et al., [No-Transaction Band Network](https://arxiv.org/abs/2103.01775v1) | The learned band policy keeps current holdings inside a learned interval and trades toward its boundary outside the interval. |
| Maggiolo et al., [Deep Hedging Under Non-Convexity](https://arxiv.org/abs/2510.01874v2) | Motivation for examining discontinuous execution costs and the limitations of gradient-based optimization. The fee presets here are synthetic research assumptions. |
| Schmid and Oeltz, [Towards a fast and robust deep hedging approach](https://arxiv.org/abs/2504.16436v1) | Motivation for studying adjustment to changed market parameters. The A → B → A helper supplies an evaluation sequence, not their embedding architecture. |

The Heston variance/stock step adapts the Andersen quadratic-exponential
construction used in [PFHedge's Heston implementation at commit `1fc08c7`](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/pfhedge/stochastic/heston.py).
It is an approximate time discretization without an exact martingale correction.
The learned band adapter also draws on
[PFHedge's pinned no-transaction-band example](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/README.md).

QuantLib 1.43 supplies independent numerical references, including its
[analytic Heston engine](https://github.com/lballabio/QuantLib/blob/v1.43/ql/pricingengines/vanilla/analytichestonengine.cpp)
and [Bates process](https://github.com/lballabio/QuantLib/blob/v1.43/ql/processes/batesprocess.cpp).
Reference agreement is evidence for the checked states and tolerances; the
[validation guide](validation.md) describes its limits.
