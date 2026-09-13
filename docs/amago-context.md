# AMAGO memory-based hedging

The [AMAGO adapter](../src/hedging_gym/baselines/amago.py) connects the official
memory-based learner to the common tensor hedging environment. AMAGO owns the
actor, critic, attention, replay and checkpointing. The adapter supplies book
sequences, action coordinates and batched evaluation.

## Sources and environment

The sources are [AMAGO](https://arxiv.org/abs/2310.09971),
[AMAGO-2](https://arxiv.org/abs/2411.11188), and
[official code pinned at 54c25ab](https://github.com/UT-Austin-RPL/amago/tree/54c25ab6da9371614c47352f569a56c0fe938d3b)
(version 3.4.0). The `multitask` switch selects the official AMAGO-2 agent.
This integration does not reproduce every published training setting.

AMAGO is optional and is not installed with Hedging Gym. The pinned native
package declares Gymnasium at most 0.29.1, whereas the finance environment uses
Gymnasium 1.2.3. Use separate native and finance environments; the finance
environment requires an explicit dependency override. A normal combined
installation cannot satisfy both declarations. The adapter exposes two
exploration-wrapper properties removed by Gymnasium's implicit forwarding.
Both runners select `VanillaAttention` and do not require FlashAttention.

Verify the installed donor commit yourself: the AMAGO runners record the
documented pin but do not validate the installed source version.

## Financial contract

A training sequence contains three independent, completely settled books from
one observed market. Cash and positions reset between books; learner memory
persists until the sequence ends. Banks may differ only in market parameters.
Use finite holding bounds and pre-maturity trading. The continuous actor does
not enforce trade lots or minimum-order sizes.

The finance runner is intended for `risk.objective="es"` source banks.
With discount one and a fixed threshold taken from a source-trained DH
checkpoint, terminal negative RU rewards optimize expected RU at that threshold.
They do not jointly optimize the threshold and policy for ES. An MSE bank would
instead produce squared-loss rewards while the runner's reporting still describes
fixed RU; use the documented ES configuration for this recipe.

Declared market parameters remain observable. Completed training books may
prime query memory, but query outcomes never become context for other queries.
The supplied recipe uses two source markets; it is not a broad meta-training
distribution.

## Required finance artifacts

The `--source` directory must contain:

- `train_bank.pt` and `adaptation/source-nearby.pt` for the two source markets.
- `development_bank.pt` for evaluation.
- `policies/dh-seedN/latest.pt` matching `--seed N`, including its trained
  threshold and policy configuration.

Saved banks must contain serialized `config`, `spot`, `variance`, `marks` and
`liability` fields, with their generation seeds recorded separately or alongside them.
These files are required even with `--load-weights`. Generate and train them
with the matching current configuration before using this runner.

## Run

Run from the repository root with the Python executable from the appropriate
optional environment; replace every placeholder:

```bash
PYTHONPATH=src:. /path/to/native-amago/bin/python -m benchmarks.qualify_amago \
  --output /path/to/native-run --epochs 3 --batches 10
PYTHONPATH=src:. /path/to/finance-amago/bin/python -m benchmarks.qualify_amago_hedging \
  --source /path/to/qualification --output /path/to/amago-hedging \
  --seed 7 --epochs 10 --batches 100 --vectorized-actors 128
```

The native task is official 5×5 MetaFrozenLake with ten attempts. Its map
generator does not receive a seed, so global seeds do not fully determine maps.
These commands check a transfer recipe, not paper-table reproduction.

For finance, `--traj-encoder ff` selects the official memory-free encoder and
`--agent-type multitask` selects AMAGO-2. `--context-review` compares reset
memory, current-market context, previous-market context and a time-index-matched
reset control on common A→B→A queries. This primes a new controller per stage;
it is not a single persistent forgetting experiment.

`--load-weights /path/to/policy_epoch_N.pt` skips training and loads an AMAGO
policy-weight file, not a training-state directory. Match the agent type,
encoder and financial schema. It does not resume interrupted training.

## Outputs and limitations

Choose a distinct external output directory for each recipe. Outputs contain
native settings, checkpoints, replay, work counts and evaluation tapes.
Completed books can reuse training paths and are not counts of unique paths.
Vector actors share one source-market draw per sequence.

Count source training, context generation and inference when comparing costs.
Native optimizer/RNG checkpoints do not guarantee exact continuation of an
interrupted environment stream. Reset-memory controls also reset previous
action/reward inputs; a claim specifically about hidden memory needs those
inputs and time indices controlled separately.
