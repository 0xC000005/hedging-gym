# Baseline implementation checklist

Implement the agreed comparison set on the existing batched environment. This
work adds method adapters, not a second simulator or accounting implementation.

- [x] Inspect existing adapters and donor implementations; assign isolated worktrees.
- [x] Preserve and run classical delta/bands, delta-gamma and delta-variance controls.
- [x] Preserve and run direct Deep Hedging and learned no-transaction bands.
- [x] Add Hull-style model-free actor-critic and EX-DRL tail-distribution adapters.
- [x] Add online fine-tuning and shared-network/task-embedding adaptive Deep Hedging.
- [x] Add learned policy/value Monte Carlo tree search with stochastic market transitions.
- [x] Add unguided sampling search and the hybrid discrete/continuous policy, with and without search.
- [x] Run every method through the common evaluator; retain configurations, checkpoints and trade tapes outside Git.
- [x] Check accounting, causal information use and the method-specific training mechanisms.
- [x] Document source-to-adapter changes and remaining competitive-training work.
- [x] Review changes, run focused/full checks, and consolidate into one task-branch commit.

Start with basic Heston, identical portfolio/observations/accounting and pooled
terminal expected shortfall. Operational frictions and market-only A→B→A remain
separate configurations. Small development runs establish that an adapter works;
they do not establish convergence, reproduce paper tables, or rank algorithms.
Report training, planning and bank-generation costs separately. Training never
uses evaluation paths. Main-branch integration follows review of the completed work.

## Next scientific step

Qualify competitive training before ranking methods: the RL critics currently
produce weak hedges, AlphaZero still uses a small action/search budget, and the
short all-method run has too few tail observations for performance conclusions.
Use development data for training choices, then fresh evaluation paths and
multiple training seeds. Do not declare a winner from the integration run.
See [method sources, APIs and commands](../methods/README.md).

## Integration outcome

All 15 controller variants completed the 30-date basic Heston GPU example and a
short CPU fixed-fee example. Both adaptive methods completed market-only A→B→A.
All outputs were finite, all executed trades were legal, and independently
reconstructed cash matched the saved tapes within floating-point tolerance.
These are adapter checks, not published-result reproductions or a ranking.
The full suite has 79 tests; the environment/pricing code and dependencies are
unchanged by this implementation.
