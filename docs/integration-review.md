# Local integration review — 2026-09-12

This is a code/evidence preservation checkpoint, not a new scientific result.
The review branch is based on local main `70faa144`; main merge, push and
worktree deletion remain held. The annotated tag
`evidence-2026-09-12-hedging-closeout` retains the assembly ancestry below.
Original branches, linked checkouts and external run artifacts remain intact.

## Retained branch coverage

| Original branch tip | Maintained code retained |
|---|---|
| `feat/baseline-suite` · `433b5ff` | Common financial core, configured paper tasks, baseline suite and separate source-loop AlphaZero bridge |
| `feat/advanced-offpolicy` · `00afb506` | `methods/off_policy.py`, qualification runner and replay/continuation checks |
| `feat/simbav2-baseline` · `ad7fe1d3` | `methods/simbav2.py`, optional official-donor runner/tests |
| `feat/hull-ddpg-qualification` · `70e32b3c` | `methods/hull_ddpg.py`, distinct 2021 two-moment learner and runner |
| `feat/r2-fast-adaptation` · `05fc719f` | Fast-adaptation comparisons, `methods/meta_pretraining.py`, attribution and search diagnostics |
| `feat/r2-geps-adaptation` · `c64b8c31` | `methods/geps.py`, source-parity tests and attribution notes |
| `feat/r2-srsa-adaptation` · `689498e9` | `methods/skill_retrieval.py` and transfer-risk tests |
| `feat/r2-belief-adaptation` · `873b67c3` | `methods/belief_adaptation.py`, encoder tests, source/license notes |
| `feat/r2-counterfactual-update` · `549d17a2` | `methods/counterfactual.py` and qualification/review runners; already ancestor of three-route work |
| `feat/r2-three-route-adaptation` · `54e21a42` | `methods/amago_adapter.py`, optional qualification runners/tests |
| `feat/r2-three-route-counterfactual` · `e6878090` | `methods/joint_counterfactual.py`, matched diagnosis and confirmation runners |
| `feat/r2-three-route-curriculum` · `d95f3b72` | `methods/curriculum.py`, uniform/regret/pooled-RU controls and qualification runner |

All tips are ancestors of the assembly tag. Overlapping GEPS/retrieval/belief
patches were reconciled against their later fast-adaptation versions; no second
active copy was imported. Earlier implementations remain in tag ancestry rather
than alternate active modules. No unique branch is left outside that ancestry.
External donor checkouts, environments, checkpoints, raw tapes and dated results
are deliberately not vendored. No old Thesis-Experiments core was imported.

## Compatibility and scientific limits

`src/hedging_gym/` is unchanged from the preserved baseline-suite tip. Learners
stay in attributed `methods/` modules; runners use that one financial ledger.
Historical adaptation runners load `experiments/configs/legacy-basic-heston.json`
(plain QE, old coefficients, stock/call book, 30/252 clock, ES95). Current paper
presets and QE-M defaults are different tasks, not updates to old evidence.

Two small checkpoint helpers reconstruct actually saved dataclass fields and
decode old missing scheme fields as QE. Optional-field additions no longer fail
equivalent-task comparisons; real scheme/config changes still fail. Retrieval
keeps its original eight numeric market features, treating scheme categorically.
Experimental contracts are explicit: embedding and SimBa policies need finite
bounds; meta/curriculum/CrossQ/TQC/SimBa training is ES-specific; AMAGO and
counterfactual branches reject maturity-date trading. No new algorithm recipe
was introduced. Some optional donor runtimes and historical runner duplication
remain; this closeout does not attempt a framework or dependency overhaul.

See [current qualification status](baseline-implementation.md) for completed,
weak and negative results. Adaptive DH is implemented; conditional Deep Bellman
Hedging is not. Stronger ordinary pretraining is a stronger baseline, not novelty.

## Validation performed

- Locked baseline dependencies synchronized; full practical suite with pinned
  GEPS donor: **175 passed, 2 optional skips**, four expected Gym/PPO warnings.
  Command: `uv run --frozen --no-sync --group baselines pytest -q -p no:cacheprovider`,
  with `PYTHONDONTWRITEBYTECODE=1`, two OpenMP/MKL threads and `GEPS_DONOR_ROOT`.
- Optional retained runtimes: SimBaV2 **3 passed** and AMAGO **4 passed**, explicitly
  importing this checkout. Final checkpoint compatibility regression rerun:
  **20 passed**. Ten new test functions address integration defects/unsupported
  contracts; most suite coverage already existed on the integrated branches.
- Existing files strictly loaded: 15 original/ordinary/meta/GEPS/belief pretrained
  policies, six ordinary/meta latest policies, three encoders, three retrievers
  and 21 full adaptation states. Ordered source configs match saved QE banks;
  updater policy, optimizer and minibatch RNG states match exactly.
- Seed-7 CrossQ/TQC `phase13` model/replay/runtime, Hull2021 `latest.pt` learner
  and optimizers, and SimBaV2 `checkpoint.pt` serialized learner/optimizer/RNG
  loaded from their existing run directories. These reads performed no training,
  evaluation, path generation or artifact writes. Tiny existing unit-test learner
  loops are implementation checks, not new scientific runs.

Checkpoint paths are under `/home/max/Documents/hedging-gym-runs/`:
`r2-fast-adaptation-2026-09-08-DChV47/comparison/`,
`r2-adapt-aware-2026-09-08-II2T94/`,
`advanced-rl-2026-09-07-JtfPLE/{crossq,tqc,simba}-100k-seed7/`, and
`basic-comparison-2026-09-07-h7WXZD/hull2021/resumed-seed7/`.
Loading compatibility and test continuation do not certify convergence,
historical performance equivalence under changed settings, or publication readiness.
