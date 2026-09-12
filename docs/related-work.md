# Related work

These references explain the research ideas and numerical constructions used
here. The benchmark and method adapters make their own disclosed choices;
they do not reproduce the cited papers' complete experiments.

| Reference | Connection to this repository |
|---|---|
| Bühler, Gonon, Teichmann and Wood, [Deep Hedging](https://arxiv.org/abs/1802.03042v1) | Learning constrained hedging strategies under transaction costs and a terminal risk objective. The benchmark uses a tradable European option as an additional hedge. |
| Imaki et al., [No-Transaction Band Network](https://arxiv.org/abs/2103.01775v1) | The learned band policy keeps current holdings inside a learned interval and trades toward its boundary outside the interval. |
| Maggiolo et al., [Deep Hedging Under Non-Convexity](https://arxiv.org/abs/2510.01874v2) | Motivation for examining discontinuous execution costs and the limitations of gradient-based optimization. The fee presets here are synthetic research assumptions. |
| Cao, Chen, Hull and Poulos, [Deep Hedging of Derivatives Using Reinforcement Learning](https://ssrn.com/abstract=3514586), [author code](https://github.com/rotmanfinhub/deep-hedging-research/tree/b4d031a185fe2547dd81ad7a67081f6dbe52c5bc) | The original two-moment DDPG hedger: critics estimate the first and second moments of future hedging cost. This is distinct from the later quantile-based learner below. |
| Cao et al., [Gamma and Vega Hedging Using Deep Distributional Reinforcement Learning](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1129370/full), [author code](https://github.com/rotmanfinhub/gamma-vega-rl-hedging/tree/77dc48326da000d983b1fb750edb2177e38c75fd) | QR-D4PG estimates a conditional return distribution for an option-arrival book with automatic stock delta hedging. The common task instead lets the policy choose stock and option holdings jointly. |
| Malekzadeh et al., [EX-DRL](https://arxiv.org/abs/2408.12446), [author code](https://github.com/pmalekzadeh/EX-DRL/tree/f1abe99df7fa9efaa65af6b9dd416c3425c64098) | Extends quantile distributional RL with a generalized Pareto tail. Native VaR/CVaR objectives must be distinguished from the common benchmark's reported terminal ES. |
| Szehr, [Hedging of Financial Derivative Contracts via Monte Carlo Tree Search](https://arxiv.org/abs/2102.06274), [author code](https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba) | Policy/value-guided search and visit-target training in trinomial, GBM and Heston environments. This release is not the complete implementation of the later non-convex comparison by Maggiolo et al. |
| Schmid and Oeltz, [Towards a fast and robust deep hedging approach](https://arxiv.org/abs/2504.16436v1) | `methods/adaptation.py` adapts the shared-network/task-embedding mechanism: fit source-market embeddings jointly, then fit a new embedding with shared weights frozen. The A → B → A sequence is our separate adaptation/forgetting evaluation. |
| Alvo, Russo and Kanoria, [Hybrid Policy Optimization](https://arxiv.org/abs/2605.14297), [author code](https://github.com/MatiasAlvo/hybrid-rl/tree/e48ae86da1e8f14c93cbb56e48d87f8674228659) | Combines discrete-action score gradients with differentiable continuous decisions. Our adapter uses HOLD/TRADE modes and differentiable sizing through the same financial ledger. |
| Bhatt et al., [CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity](https://arxiv.org/abs/1902.05605v4), ICLR 2024; [author code](https://github.com/adityab/CrossQ) | Continuous-control baseline using batch normalization and no target networks. We use the maintained SB3-Contrib implementation; its native continuous-control results are motivation, not evidence of better hedging. |
| Kuznetsov et al., [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://proceedings.mlr.press/v119/kuznetsov20a.html), ICML 2020; [SB3-Contrib TQC](https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html) | TQC truncates an ensemble of quantile-critic predictions to control value overestimation. It is a separate baseline from CrossQ. Its distributional critic does not by itself make the training objective expected shortfall. |
| Lee et al., [Hyperspherical Normalization for Scalable Deep Reinforcement Learning](https://arxiv.org/abs/2502.15280v2), ICML 2025 Spotlight; [author code](https://github.com/DAVIAN-Robotics/SimbaV2) | SimBaV2 stabilizes SAC-style continuous-control training through normalized representations/weights and distributional value estimation. We retain the official learner and bridge it to the common financial environment. This is not the separate contrastive-RL depth-scaling paper. |

The Heston variance/stock step uses Andersen's quadratic-exponential construction.
The default `scheme="qe_m"` adds the conditional stock martingale correction in
[QuantLib 1.43's `QuadraticExponentialMartingale` implementation](https://github.com/lballabio/QuantLib/blob/v1.43/ql/processes/hestonprocess.cpp).
Plain `scheme="qe"` retains the uncorrected construction adapted from
[PFHedge's Heston implementation at commit `1fc08c7`](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/pfhedge/stochastic/heston.py).
Both are approximate time discretizations; the martingale correction does not
make Heston joint-path simulation exact. Bates uses the selected Heston scheme
for its diffusion component.
The learned band adapter also draws on
[PFHedge's pinned no-transaction-band example](https://github.com/pfnet-research/pfhedge/blob/1fc08c73756bc6350f6a66977a5be97497d3bca0/README.md).

QuantLib 1.43 supplies independent numerical references, including its
[analytic Heston engine](https://github.com/lballabio/QuantLib/blob/v1.43/ql/pricingengines/vanilla/analytichestonengine.cpp)
and [Bates process](https://github.com/lballabio/QuantLib/blob/v1.43/ql/processes/batesprocess.cpp).
Reference agreement is evidence for the checked states and tolerances; the
[validation guide](validation.md) describes its limits.

The [implementation appendix](baseline-methods.md) separates each source method,
its common-task changes, and the checks needed before interpreting its results.
