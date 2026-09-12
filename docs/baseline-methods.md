# Baseline implementation appendix

The comparison has two different purposes. An **author-code reference run**
checks our understanding on the author's own task. A **common-Heston transfer**
compares the retained learning mechanisms in one financial environment. Neither
is presented as an exact reproduction of every paper table.

## One financial task

The basic comparison uses Heston with stock, one longer-dated European call and
cash to hedge a short European call. There are 30 daily decisions, proportional
trading costs and holding limits; no fixed fees, lots or regime changes. This
follows the simulated, self-financing design of Deep Hedging, but using a
tradable call rather than the original paper's variance-swap hedge is a disclosed
benchmark choice.

All controllers receive the same causal observations and start with the same
capital. Targets enter the same ledger, including trading costs and final
liquidation. Discretized methods must disclose their available holding grid;
continuous methods are not silently rounded to that grid. Market paths used for
search are sampled conditionally, not read from the realized evaluation future.

The primary reported metric is upper-tail ES95 of terminal loss. Mean loss,
transaction costs, turnover, training work and decision-time work are reported
alongside it. Results across training seeds measure optimizer variability;
resampling evaluation paths alone does not do so.

## Method mapping

Full references and pinned source links are in [related work](related-work.md).
The table describes maintained method modules; historical author-learner
transfers are identified separately below.

| Method | Mechanism retained | Common-task change and implementation boundary |
|---|---|---|
| Delta / delta-gamma | Current-state model sensitivities determine target holdings | Heston sensitivities use the shared pricer and configured holding limits; they do not optimize ES. Ordinary delta has no added no-trade band. |
| Deep Hedging | Differentiate terminal portfolio loss through causal trading decisions | PyTorch policy and common accounting; bounded stock/call targets and joint ES-threshold fitting. Market simulation does not expose future observations to the policy. |
| No-transaction bands | Learn an interval of acceptable holdings; trade to its boundary when outside | One interval per instrument, with learned centers and widths. This is a band architecture, not ordinary delta hedging. |
| Cao/Hull 2021 DDPG | Prioritized replay, target actor/critics, first- and second-moment Bellman targets, actor improvement through critics | Joint stock/call actions replace stock-only trading. The source mean-plus-standard-deviation objective is distinct from ES. Any source-code versus paper-equation corrections are disclosed separately. |
| QR-D4PG (`qr_d4pg.py`) | Quantile critic and deterministic actor gradients through estimated terminal risk | Maintained PyTorch port with uniform replay, Polyak targets and a global terminal-ES objective. Joint holdings replace the source option action and automatic delta hedge. |
| EX-D4PG (`exdrl.py`) | Quantile critic with a fitted generalized Pareto tail in targets and actor improvement | Maintained PyTorch port with the same global-risk collector; inverse-CDF quadrature and analytic tail expectations replace sampled tail integration. |
| Adaptive Deep Hedging | Shared network and source-task embeddings; adapt a new embedding with shared weights frozen | Paper-mechanism implementation with common observations, configured holdings and terminal risk objective. Full-network fine-tuning is a separate control. |
| AlphaZero | Network-guided search, visit-distribution policy targets and realized-return value targets | `alphazero.py` is the maintained ES/reanalysis adaptation. `source_alphazero.py` imports Szehr's search/training and deploys policy argmax during evaluation. They are distinct implementations. |
| SB3 PPO | The installed PPO learner, without changing its update equations | A thin batched environment maps normalized actions to holdings and supplies the declared complete-episode reward. This is an independent RL control, not an implementation of the Hull or EX papers. |
| CrossQ | SB3-Contrib's batch-normalized SAC-style learner without target networks | Continuous stock/call holdings and a terminal global-risk reward. Replay rewards must use the current training-only ES threshold, not the threshold stored when a path was collected. |
| TQC | SB3-Contrib's truncated ensemble of continuous quantile critics | Same financial interface and global-risk reward as CrossQ. Quantile truncation addresses critic overestimation; the reward mapping, not truncation, supplies the ES objective. |
| SimBaV2 | Official JAX learner: hyperspherical normalization and SAC-style distributional value learning | Thin observation/action/reward bridge, with the same financial ledger and current-threshold replay treatment. No new market model or discretized action grid. |
| Hybrid policy optimization | Discrete score-function gradients plus differentiable continuous sizing | HOLD/TRADE modes and shared cost-inclusive terminal risk; subsequent categorical PPO updates use detached histories. No straight-through derivative is substituted for the discrete choice. |
| CEM / guided rollout search | Sample candidates, evaluate conditional rollouts, refit to elites | Root-action improvement with frozen-policy feedback is a rollout-planning baseline, not full AlphaZero. Gradient refinement and learned guidance are separately costed ablations. |

## Training objectives are not interchangeable

Deep Hedging and the common global-risk adapters minimize the sample mean of

$$
\zeta + \frac{(L-\zeta)^+}{1-\alpha},
$$

jointly with one episode-global threshold $\zeta$. The shared evaluator reports
empirical terminal ES after pooling complete paths, not an average of minibatch
ES estimates.

The original 2021 actor instead uses critics for conditional first and second
moments. The original 2023 and EX learners use conditional distributional risk.
Reporting all their policies' terminal ES does **not** make their training
objectives identical. Preserve original-objective rows and identify any separate
common-objective variant; do not claim an algorithm is inherently inferior from
a comparison that changes both its objective and implementation.

The maintained `qr_d4pg.py` and `exdrl.py` modules are global-risk PyTorch
ports. Historical `hull_rl` artifact keys refer to the QR-D4PG port, not
Hull's separate two-moment DDPG. The original TensorFlow QR-D4PG and EX-DRL
learners were also transferred in earlier experiments with native conditional
risk objectives. Those [author-learner results](baseline-implementation.md)
remain separate evidence; they do not qualify the maintained ports.

## What makes an implementation defensible

- Map the cited algorithm to its actual update, loss, action representation and
  training recipe. Native code, a paper-mechanism reimplementation and a
  common-task adaptation receive different labels.
- For changed calculations, compare fixed-input outputs, targets or gradients
  with the source. Check saved-policy reloads and independently reconstruct the
  executed cash ledger. A passing API test alone is insufficient.
- Train on declared data/budgets, retain learning curves and checkpoints, and
  investigate unexplained collapse before using it as evidence against a
  method. Source recipes may differ in budget; report those differences.
- Freeze policies before fresh final evaluation. Compare identical paths and
  initial capital; report seed variation and complete compute. Preserve poor
  results without attributing them to the broader algorithm prematurely.

Batching, GPU execution and numerical streaming are implementation choices.
Their checks establish the numerical scope of the change; equal seeds alone do
not prove identical CPU/GPU random samples or bit-identical optimizer histories.
