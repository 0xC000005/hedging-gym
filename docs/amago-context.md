# Context-aware hedging qualification

This adapter asks whether an official memory-based learner can reuse experience
from completed books. It does not change the market simulator, portfolio, costs,
observations or terminal accounting. Market parameters remain observed, so memory
does not reveal a hidden regime: its possible benefit is learning or transferring
a useful strategy more efficiently.

## Borrowed method

[AMAGO](https://arxiv.org/abs/2310.09971) trains an off-policy actor and critic on
long Transformer sequences of observations, previous actions and rewards.
[AMAGO-2](https://arxiv.org/abs/2411.11188) changes critic targets to classification
and uses a filtered policy update to improve training across different return
scales. Both run through the
[official code](https://github.com/UT-Austin-RPL/amago/tree/54c25ab6da9371614c47352f569a56c0fe938d3b).
This is source version 3.4.0, not a frozen reproduction of every paper setting.

`src/hedging_gym/baselines/amago.py` supplies only the financial environment
interface and evaluation controller. Actor/critic learning, replay, attention and checkpointing
come from AMAGO. The `multitask` agent switch selects the official AMAGO-2 class;
it is not a locally invented optimizer. The optional dependency lives in an
isolated environment; installing the hedging library does not install AMAGO.

## Financial objective and sequence

Each training sequence contains three independent, completely settled books
from the same observed market. The shared tensor environment resets cash and
positions between books. The learner's memory persists until the sequence ends.
The batched collector uses the same ledger as the scalar collector.

AMAGO optimizes expected return. With discount one and terminal reward equal to
negative Rockafellar–Uryasev loss at a **fixed** threshold, it optimizes expected
RU loss at that threshold. The threshold is taken from a source-trained DH
checkpoint, never fitted on evaluation outcomes. This preserves replay reward
consistency, but is not joint threshold/policy optimization for expected shortfall.
Evaluation reports actual terminal ES and RU separately.

The current qualification uses two source markets, not a broad meta-training
distribution. All source and query paths are exogenous. Completed training books
can prime evaluation memory; query outcomes cannot prime other queries.

## Running the checks

Use an isolated environment with the pinned AMAGO source and its dependencies.
The native example uses AMAGO's Gymnasium 0.29.1 dependency. The financial adapter
needs the common environment's Gymnasium 1.2.3; two explicit exploration-wrapper
properties replace the forwarding removed by Gymnasium. No author file is edited.

From this checkout, using that environment's Python:

```bash
PYTHONPATH=src:. python -m benchmarks.qualify_amago --output /path/to/native-run --epochs 30 --batches 100

PYTHONPATH=src:. python -m benchmarks.qualify_amago_hedging --source /path/to/qualification-banks --output /path/to/finance-run --epochs 20 --batches 100 --vectorized-actors 128
```

The native task is the official 5×5 MetaFrozenLake with ten attempts. A shortened
learning run establishes source execution, not a paper-table reproduction.
The pinned source generates maps without passing a seed, so the script's global
seeds do not make native task generation completely reproducible.

For finance, `--traj-encoder ff` selects AMAGO's own memory-free encoder;
`--agent-type multitask` selects AMAGO-2. The source directory supplies existing
`train_bank.pt`, `adaptation/source-nearby.pt`, `development_bank.pt` and trained
DH checkpoints under `policies/dh-seedN/latest.pt`. The ordinary adaptation review
also uses existing `adaptation/seed-N/multitask-policy.pt` checkpoints.

`--context-review` compares reset memory, current-market context and previous-market
context on identical A→B→A query paths. It also starts an empty memory at the same
later time index: an apparent context gain could otherwise be only a sequence-
position effect. `--load-weights` runs this review on an existing native checkpoint
without retraining. Context-path generation and inference are additional work,
not free adaptation.

This context screen primes a new controller with two completed books for each
stage; it is not a single persistent A→B→A forgetting experiment. Resetting also
zeros the previous action/reward input. A positive memory claim would require
holding those inputs and the time index fixed while resetting only hidden memory.
The separate DH/embedding review does carry learned weights through A→B→A.

All banks, tapes, replay, settings and checkpoints go to the external output
directory. Reports count actual updates, completed books and fixed-threshold
exceedances. Completed books can reuse training paths and are not a count of
unique independent paths. Checkpoints retain native optimizer and RNG state;
the scripts do not promise bit-identical interrupted environment continuation.
The current vector collector selects one source market shared by all actors in
a sequence. Twenty epochs give twenty completed task draws, not 7,680 distinct
meta-tasks, even though the individual training books have different sampled paths.

## What would constitute evidence?

A useful memory result must beat a strong parameter-conditioned control and a
time-index-matched reset-memory control, not merely an untrained policy. Compare
training objectives and budgets explicitly. Inspect ordinary DH fine-tuning and
task-embedding adaptation on the same A→B→A markets before attributing an effect
to the new donor. A poorly trained critic, weak meta-training distribution or
fixed-RU mismatch can reject this transfer recipe without rejecting adaptation
as a research direction.
