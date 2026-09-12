# Hedging Gym

This private research repository owns the financial environment, baseline
methods and paper benchmarks in one `hedging-gym` package.

## Structure and scientific contract

- `src/hedging_gym/environment/`: market simulation, pricing, one cash ledger
  and Gym/tensor interfaces, differentiable episodes and conditional branches.
  This layer must not import baseline learners or choose a search action.
- `src/hedging_gym/interfaces.py`: public `Controller` decision contract.
- `src/hedging_gym/adapters/`: external-library environment translation only.
  For example, `sb3.py` handles VecEnv conventions, not the SB3 learner itself.
- `src/hedging_gym/baselines/`: one named module per method, with paper and
  upstream source links in its header. Reuse author code or established learners;
  document deliberate source changes. Configuration variants share a module.
- Controllers follow `interfaces.Controller`: current observations and ledger
  in, batched target holdings out. Only the environment executes trades. Keep
  training method-specific; do not require dummy trainers or a common framework.
- `baselines/_shared/` contains reused learner mechanics: pathwise training,
  adaptation updates, replay, network blocks and checkpoint handling. Replay
  reward relabeling belongs here, not in a library-format adapter.
- `extensions/` contains additions to existing policies, such as retrieval,
  curriculum and counterfactual training. Do not move complete methods here
  merely because they are adaptive or experimental.
- Use public environment stepping, `rollout.run_episode`, planning operations
  or financial primitives as appropriate. Preserve autograd and caller-owned
  network mode during training; planning samples fresh conditional futures.
  Keep algorithm-specific collectors when replay/search needs them. Do not
  duplicate accounting, force a universal trainer or add empty forwarding files.
- `src/hedging_gym/evaluation.py`: common terminal-risk evaluation.
- `benchmarks/`: runnable comparisons and paper configurations in `configs/`.
  `tests/` checks financial/API/baseline behavior; `docs/baselines.md` maps methods.
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
- Formal comparisons use fresh paths, multiple training seeds, the declared
  pooled terminal cost-inclusive objective (such as MSE or ES) and complete
  compute costs. Declare configurations and
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
- Do not add legacy checkpoint compatibility layers: current checkpoints must
  match explicit configuration; historical runs use archived source and artifacts.

## Git

- After the initial import, work on a task branch; use isolated worktrees for
  independent parallel edits. Reserve file ownership before parallel work.
- Stage explicit files, review the diff and make one coherent commit per change.
  Squash fixups at integration; do not import the old experiment commit history.
- The user approved the 2026-09-12 closeout: integrate the reviewed squash,
  push main and the evidence tag, then remove only the audited stale worktrees
  and branches after verifying their remote preservation.
- Push only to the intended private remote. Subsequent main merges need user
  approval. Preserve unrelated edits and historical research evidence.
- Remove a completed worktree/branch only after its retained work is integrated;
  a worktree is not an archive. Preserve historical research evidence.
