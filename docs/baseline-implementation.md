# Baseline implementation and qualification

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
