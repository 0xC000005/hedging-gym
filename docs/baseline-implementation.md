# Baseline implementation and qualification

## Active objective: defensible basic-Heston comparison

Qualify the intended literature baselines on the existing basic Heston task,
then compare frozen policies on one fresh common test bank. Do not expand the
benchmark or tune a proposed winner. Preserve completed native references and
working adapters; poor scores alone do not diagnose implementation errors.

- [ ] Finish the running native AlphaZero reference and use its evidence to guide the common-task transfer.
- [ ] Qualify the original EX-DRL learner, then transfer it without the earlier undocumented method changes.
- [ ] Add the distinct Hull/Cao 2021 two-moment DDPG method to the common task, using the completed native reference.
- [ ] Qualify acceleration for each baseline: batch independent work, measure CPU/GPU throughput, and check reference-versus-accelerated outputs before extended training.
- [ ] Complete missing source-supported training and seed checks; distinguish original objectives from common-objective variants.
- [ ] Evaluate the retained baseline set on the same fresh basic bank, including ordinary delta hedging.
- [ ] Write one source-to-implementation appendix and review code/results; keep merge and push held.

New artifacts: `/home/max/Documents/hedging-gym-runs/basic-comparison-2026-09-07-h7WXZD/`.
Independent tasks use separate environments/artifact directories and an isolated
worktree when editing repository code. Runtime pilots and progress output guide
compute decisions; they do not select scientific settings from test scores.
Keep source citations, commands, settings, checkpoints and compact result notes.
No new registry, testing framework or benchmark infrastructure is needed.

### Approved continuous-control additions

CrossQ, TQC and SimBaV2 join the existing basic-Heston comparison. CrossQ and TQC
are separate algorithms, not a combined method. Reuse SB3-Contrib for the first
two and official SimBaV2 code for the third. Rainbow is excluded from this round.

- [x] Add all three methods and original papers to the baseline appendix and related-work list.
- [x] Add minimal adapters with correct action scaling and current-threshold replay rewards.
- [x] Verify short training, saved-state recovery and GPU throughput; source update ratios are retained.
- [ ] Train with progress logs and checkpoints; compare frozen policies using the existing common evaluator.

New artifacts and the execution checklist live in
`/home/max/Documents/hedging-gym-runs/advanced-rl-2026-09-07-JtfPLE/`.
These additions do not restart or interrupt the existing recovered experiments.
Training budgets and update-to-data ratios must be reported; a fast but
undertrained pilot is not evidence against an algorithm. Merge and push stay held.

The first approved training block uses 99,840 transitions for each method and
each of seeds 7, 17 and 29, on the unchanged training/development banks. This is
an initial comparison, not an equal-training-budget ranking against the longer
PPO runs. The external checklist records source settings, warmup differences,
GPU timings and exact launch commands. No final test data selects the recipe.

The [source-to-code appendix](baseline-methods.md) is drafted. Original QR-D4PG
common-Heston seeds 17 and 29 now repeat the qualified seed-7 recipe, with
separate logs and checkpoints. EX-DRL and the original two-moment DDPG have
independent owners; AlphaZero's longer reference run does not block them.

Acceleration is part of each adapter, not a separate research method. Keep
objectives, updates per transition, replay rules and search budgets unchanged
when comparing implementations; record collection-order changes explicitly.
Use GPU batches for tensor work and CPU parallelism for independent work where
measured throughput improves. Keep progress and checkpoints. Do not rewrite a
donor merely to fill the GPU, or interrupt the running native AlphaZero anchor.

## Current work: author-code baseline anchors

Before further common-task tuning, establish the author implementations on
their original tasks. Our unsuccessful ports do not refute their published
results. Keep legacy dependencies and raw evidence outside this repository.

- [x] Evaluate the original Cao/Chen/Hull/Poulos released weights against their matching classical hedge.
- [x] Train and evaluate one Cao et al. distributional-RL case using the author recipe.
- [x] Transfer the original 2023 conditional-CVaR learner to common Heston; keep this distinct from the global-ES port.
- [x] Identify the matching AlphaZero author release and diagnose its differences from our current adapter; no speculative rewrite.
- [ ] Run the public AlphaZero source on its released native benchmark, with fresh evaluation paths and saved policies.
- [ ] Use the native AlphaZero result to guide an incremental Heston transfer; keep market, action and objective changes separate.
- [ ] Save checkpoints, original commands, compatibility changes and results; review locally without merging or pushing.

Evidence directory: `/home/max/Documents/hedging-gym-runs/native-rl-2026-09-07-VnrGDa/`.
Native evaluation and independent training are distinct checks. The 2023
repository's bundled policy/logs are short demonstrations, not paper results.

The approved AlphaZero reference uses the released trinomial terminal-variance
task: 60 dates, 21 stock holdings, fresh market draws, original shared search
statistics and candidate acceptance. The fixed budget is 10 cycles of 1,000
self-play episodes, 25 simulations per decision and 500 validation episodes
per candidate/incumbent comparison. Ten cycles follows the earlier paper's
Figure 5.2 budget; the released 60-date/feed-forward setup is not that figure's
exact configuration. A runtime-only pilot selected CPU: its tiny-network
inference was about four times faster than GPU. Training and independent final
evaluation are tracked in `alphazero-native/`; the Heston AlphaZero adapter has
not received another learning/search recipe change. Its policy-only evaluation
now skips an unused critic calculation, with the same selected actions.

Adapters use the existing batched market, observations, legal trades and cash
ledger. The common Heston experiment adapts the source methods; it does not
reproduce their paper tables. See [methods and sources](../methods/README.md).

## Execution checklist

- [x] Implement and exercise the complete classical, DH/band, RL, adaptation, AlphaZero and hybrid/search set.
- [x] Assign isolated method worktrees and train concurrently where resources allow.
- [x] Save full training checkpoints and verify interrupted/resumed trajectories.
- [x] Generate shared training/development banks; keep final evaluation separate.
- [x] Train and diagnose every family; preserve unsuccessful runs and source-backed corrections.
- [x] Reload saved models and evaluate actual hedges, not just critic predictions.
- [x] Test fixed fees separately from market-only A→B→A, with frozen adaptation controls.
- [x] Evaluate sampled/greedy hybrid deployment and common-continuation search budgets.
- [x] Independently reconstruct saved cash ledgers and record per-method outcomes.
- [x] Finish local code review and squash qualification changes; do not merge or push.

**Merge and push remain on hold.** Completion means every method was assessed,
not that every method is a competitive comparator. Classical controls and CEM
do not train neural networks; their decisions and accounting still need checking.

## Initial qualification outcomes

| Method | Evidence | Remaining limitation |
|---|---|---|
| Delta, delta-band, delta-gamma, delta-variance | Basic/fixed-fee evaluations and independent cash checks | Sensitivity hedges do not directly optimize ES; band width remains the disclosed .05 |
| Deep Hedging | Three trained/reloaded seeds in both configurations | Not proof of convergence or universal superiority |
| Learned no-transaction bands | Three seeds; hold/trade behavior checked | Seed-sensitive; ordinary pathwise gradients miss moving fixed-fee boundaries |
| Hybrid policy | Three seeds in both configurations; sampled and greedy deployment evaluated separately | Not consistently better than DH/bands |
| Hull/Rotman RL | Source-backed corrections produced useful seed-7 learning | Single recovered seed; imperfect critic calibration |
| EX-DRL | Useful seed-7 learning, then two additional seeds tested | Only one of three learned well; not a reliable competitive comparator |
| Full-network fine-tuning | Paired frozen/adapted DH completed A→B→A | One source-policy seed; pretraining budget differs from multitask pretraining |
| Task-embedding adaptation | Three A→B→A runs; shared weights exactly frozen; forgetting/recovery measured | Not proof it beats full fine-tuning at matched total work |
| AlphaZero/MCTS | Actual self-play/visit/value learning; corrected checkpoint retained after 16 batches | Poor learning plateau; search did not rescue it; not competitively qualified |
| Unguided CEM | Causal search with trained feedback; 16/64-scenario sensitivity measured | Small scenario budgets badly misrank candidates |
| Guided/refined search | Same hybrid continuation and base candidate budget; basic/fixed-fee cases evaluated | Mixed results on 512 development paths and one policy seed; no superiority claim |

Lot sizes and minimum-order constraints remain unsupported by continuous
adapters. No silent rounding or projection was added to make them pass.

## What training exposed

- **AlphaZero action identity:** the old mask disabled an absolute grid action
  when it matched current holdings, forcing another identity for HOLD. Stable
  absolute actions now remain available, matching the source. This fixed a real
  porting issue but did not solve the learning plateau.
- **RL conditioning:** the initial port supplied far less replay training than
  the source and lacked exploratory replay warmup. Both were corrected.
  Source-style intermediate marked-PnL labels then helped learning. They
  telescope to the same terminal loss; shifting the global risk threshold
  preserves terminal ES. Independent review confirmed causal, detached labels.
  Reliable critic learning remains a separate requirement.
- **Checkpoint/configuration:** reject a requested total below the saved step;
  overlay operational fees without replacing the saved book's other execution
  settings. These bugs did not affect the completed standard-bank runs.

## Experiment scope and artifacts

Common data: 16,384 training paths (seed 1101), 8,192 separate development paths
(2201), 30 Heston dates, unchanged stock/call/cash book. Fixed-ticket fees overlay
the same market paths. DH/bands/hybrid use seeds 7, 17, 29, batch 256, hidden
layers 64/64, 3,000 basic updates and 1,000 fixed-fee updates. These are disclosed
development budgets, not convergence certificates.

Every learned family retains first-update and latest full snapshots, including
relevant optimizers, threshold, critic/targets, RNG and replay. Longer runs
replace the latest snapshot periodically; completed stages and unsuccessful
runs are also retained. Exact split/resume checks cover same-device continuation.

Artifacts and commands:
`/home/max/Documents/hedging-gym-runs/qualification-2026-09-07-TyDCMB/`.
Its README indexes checkpoints, curves, losses, source notes and independent
reconstructions. Raw data stays outside Git. The full suite passes 88 tests;
the financial environment, pricing implementation and dependencies are unchanged.

Concurrent timings are not a formal speed comparison. Search diagnostics use
512 common paths, not the 8,192-path policy bank; do not mix them into one ranking.

**Formal comparison is deferred:** resolve the RL/AlphaZero qualification gaps,
establish competitive recipes, freeze settings, then compare saved models on
fresh common paths and multiple seeds. Report basic, operational and adaptation
separately, including training and planning costs. These unsuccessful runs
reject only their tested recipes, not the broader methods or thesis direction.

## Training-signal repair

The subsequent local repair leaves the financial package unchanged. It corrects
RL action-gradient clipping and tail-fitting settings, aligns AlphaZero's global
risk threshold with its value targets, and adds an optional stock SB3 PPO control.
The same training/development banks and terminal ES95 evaluator are retained.

| Adapter | Final development ES95 | Assessment |
| --- | --- | --- |
| SB3 PPO, three seeds | Sampled: 0.014425–0.024030; greedy: 0.010566–0.017501 | All learned useful hedges; still not better than the strongest DH controls |
| QR-D4PG, three seeds | 0.150930 / 0.177547 / 0.066716 | Actor saturation repaired, but critic guidance becomes unreliable as the actor changes; all worse than their initial policies |
| EX-D4PG, one seed | 0.066612 | Tail fitting improved, but actor performance still deteriorated; not competitively qualified |
| AlphaZero, three short seeded runs | 0.273902 / 0.135041 / 0.097827 | Threshold consistency repaired; value accuracy and useful search remain unqualified |

These are diagnostic endpoints, not selected best checkpoints or an equal-budget
ranking. PPO used 512 complete episodes per rollout and 262,144 training episodes
per seed. Its improvement over the earlier small probe cannot be attributed to
batching alone: budget and training-bank reuse also changed. QR-D4PG used 200
frozen-actor warmup updates followed by 300 actor-enabled collection updates;
EX used 400 plus 300. AlphaZero used four self-play batches per seed, with
separate completed-rollout value refitting. Its short runs do not reject the
method at larger budgets. No additional tree evaluation was justified by the
inconclusive value diagnostics.

Initial critic checks are diagnostics, not arbitrary pass/fail barriers to
studying a learner. Failed checks remain recorded. The later bounded actor runs
test whether learning works despite those warnings; they do not certify the
critics. Evaluation-bank results do not decide training continuation.

The integrated suite passes 97 tests. New checks cover the changed equations,
action gradients, reward/autoreset boundary and checkpoint behavior; experiment
performance is assessed from saved runs, not asserted by unit tests.

Artifacts, exact recipes, source notes and before/after checkpoints:
`/home/max/Documents/hedging-gym-runs/repair-2026-09-07-uQK20L/`.
Saved policies reload to identical losses on the same device; all final tapes
reconcile to independent NumPy accounting within 1.2e-6. SB3 ZIP files establish
inference reload parity, not exact custom-environment training continuation.
Merge and push remain held; the weak adapters must not be used to claim a new
method beats strong implementations of the literature.

## Native source checks

The original [Cao/Chen/Hull/Poulos policy](https://github.com/rotmanfinhub/deep-hedging-research/tree/b4d031a185fe2547dd81ad7a67081f6dbe52c5bc)
was evaluated without retraining on its own GBM stock-only task. On 5,000 fresh
paired paths it improved mean cost plus 1.5 standard deviations by **16.95%**
over native delta hedging, close to the paper's 16.6%. Saved trades reconcile
to independent accounting within 5.7e-13. This verifies the released policy,
not independent training or common-Heston performance. Its objective is not ES95.

The original [2023 QR-D4PG learner](https://github.com/rotmanfinhub/gamma-vega-rl-hedging/tree/77dc48326da000d983b1fb750edb2177e38c75fd)
also completed its native GBM/2%-cost experiment: 40,000 training episodes and
5,000 fresh paired evaluation episodes. ES95 was **14.91168 versus 21.01379**
for delta-gamma, a **29.04% reduction**; the paper reports 15.37 versus 21.10.
The original learner, risk objective and financial task were preserved.
Runtime compatibility changes and progress/checkpoint additions are recorded
externally. Paired scenarios match exactly; recorded P&L sums and saved-policy
reload agree within floating-point tolerance. This successful single-seed native
run is not yet evidence for the joint-action common-Heston transfer.

### Why our AlphaZero result does not match the paper

The fetched [public author code](https://github.com/plan64/minimalHedger_AlphaZero/tree/3111c378fcd17e45f94d2fc668a3aa117126ecba)
belongs to Szehr's earlier MCTS paper, not the complete
[2025 AlphaZero versus Deep Hedging comparison](https://arxiv.org/html/2510.01874v2).
A complete official release of the latter was not located. The public repository
also explicitly says that some training details are private.

- The 2025 headline experiment used 150,000 self-play games, five decisions,
  one stock and a squared-error objective with capped costs. Our latest
  adaptation used 512 self-play games, 30 Heston dates, two traded assets and
  terminal ES95, with much smaller networks and fewer policy refits. Additional
  continuation simulations trained the critic; they do not equal self-play.
- Probing the saved checkpoints found initial searches reached only 2–4 of 30
  dates and never a terminal payoff. Their decisions therefore depended on
  learned values, which were poorly calibrated for distinguishing actions.
- The public donor reuses search statistics across self-play episodes and
  rejects worse candidate policies. Our adapter restarts every search and
  unconditionally retains updates, including observed policy deterioration.
- Our critic reanalysis evaluates the fitted greedy actor, whereas the public
  donor learns from completed search-policy trajectories. This changes the
  training loop; it is not a literal port. No new accounting or market-chance
  sampling defect was established by this audit.

These observations explain why the present result is not a faithful test of the
paper's claim. They identify likely performance causes, not measured benefits
of fixes. The next repair should establish a native anchor, restore source-backed
model selection and useful search/value training, then test the same Heston
transfer. More compute alone is not an established solution.

Detailed source locations, original commands and frozen-checkpoint probes:
`native-rl-2026-09-07-VnrGDa/alphazero-audit/` under the external runs directory.
The audit did not change AlphaZero code, checkpoints or training.
