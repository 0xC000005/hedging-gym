# Hedging Gym

This private research repository owns the common hedging environment and its
method adapters.

## Structure and scientific contract

- `src/hedging_gym/`: market simulation, pricing, one cash ledger, Gym/tensor
  interfaces and terminal-risk evaluation. Keep learners outside this package.
- `methods/`: small, explicitly attributed algorithm adapters. `experiments/`
  contains runnable comparisons; `tests/` checks financial/API behavior.
- Market dynamics, portfolio, calendar, execution rules, risk level and learner
  updates are independent choices. Compose their configurations; derive action
  sizes and observation schemas from them rather than copying benchmark sizes.
  Adaptation changes market parameters only; an operational overlay stays fixed
  through A→B→A. All methods use the same accounting,
  information, initial capital, legal trades and evaluation paths.
- QuantLib is an independent reference, not our own calculation validating
  itself. Check prices, cash accounting, discretization and policy risk separately.
  Report unresolved numerical precision honestly. API checks do not certify
  market realism or establish method superiority.
- Formal comparisons use fresh paths, multiple training seeds, pooled terminal
  cost-inclusive ES and complete compute costs. Declare configurations and
  metrics first; do not tune on final results or weaken baselines.
- Distinguish implementation checks, adaptations, reproductions and scientific
  conclusions. A narrow failure only rejects what it tested; diagnose and record
  a justified follow-up instead of inventing a convenient proxy.

## Execution and maintenance

- Reuse existing source before writing a new implementation. Keep numerical
  equations and constraints in one place; comment units and non-obvious choices.
- Use `uv sync --locked` and focused `uv run --frozen pytest` checks. Add tests for
  equations, gradients, accounting and configuration independence, not prose,
  frozen dictionaries or every hypothetical bad input.
- Print configuration, seeds, device and expected work when a run starts. Long
  loops need flushed completed/total work, elapsed time and an estimated finish.
  Batch independent work where worthwhile; don't leave an opaque serial run.
- Keep checkpoints, raw results and logs outside Git. No custom hashes,
  registries, dashboards, workflow engines or extra testing frameworks.

## Git

- After the initial import, work on a task branch; use isolated worktrees for
  independent parallel edits. Reserve file ownership before parallel work.
- Stage explicit files, review the diff and make one coherent commit per change.
  Squash fixups at integration; do not import the old experiment commit history.
- Push only to the intended private remote. Subsequent main merges need user
  approval. Preserve unrelated edits and historical research evidence.
- Remove a completed worktree/branch only after its retained work is integrated;
  a worktree is not an archive. Preserve historical research evidence.
