# Comparing fast adaptation

Can a model pretrained across several markets adapt more effectively than
Adaptive Deep Hedging? This experiment keeps the basic Heston book, observed
parameters, trading limits, proportional costs and terminal ES95 unchanged.
It changes how a pretrained hedger is conditioned on a new market.

| Method | What adapts? | Source mapping |
|---|---|---|
| Adaptive DH | Four input-context values | Shared policy and source vectors are jointly trained |
| [GEPS](geps-adaptation.md) | Four values that modulate internal weights | Low-rank context-conditioned layers |
| [SRSA-style retrieval](skill-retrieval.md) | A selected source vector, then the same four-value update | Learned transfer ranking, with optional top-five rescoring |
| [Belief encoder + DH](belief-adaptation.md) | History-inferred context, optionally updated through ES | Dynamics encoder only, not full Forward-Backward RL |

The comparison includes nearest-market and exhaustive-context selection so
learned retrieval must justify its extra work. Every method receives the same
market parameters. Each target starts independently from its pretrained model;
this is not an A-B-A forgetting experiment.

## Run

From a development installation of this repository, choose an output directory
outside Git. First generate the shared banks; CUDA accelerates option pricing:

```bash
python -m experiments.compare_fast_adaptation banks --output /path/to/run --device cuda
```

Then run each method (`adh`, `geps`, `belief`) with policy seeds 7, 17 and 29:

```bash
python -m experiments.compare_fast_adaptation compare --output /path/to/run --method adh --seed 7 --threads 1
```

Independent method/seed jobs can run concurrently. Small policy minibatches may
be faster on CPU than GPU; time a representative unit before selecting hardware.
Training emits progress and saves optimizer/RNG checkpoints. Repeating the same
command resumes pretraining and incomplete adaptation; completed curves remain
unchanged. Use a different output directory for a changed recipe or `--smoke`.

The runner fixes eight source markets, three unseen target markets and one
seen reference market. It uses 6,000 source updates and evaluates new-market
adaptation at 0, 10, 50 and 200 updates. The constants and disjoint seed families
are in `experiments/compare_fast_adaptation.py`.

```bash
python -m experiments.summarize_fast_adaptation /path/to/run
```

The summary reconstructs ES from saved loss tapes and gives exploratory paired
intervals. Three training seeds are an initial screen, not a definitive ranking.
Compare training, context history, retrieval scoring and adaptation work—not
just the number of gradient updates. Concurrent wall times are not isolated
algorithm speed measurements. Donor pretraining and source differences are
recorded in each method's documentation and output metadata.

## Does freezing the shared network help?

Reuse the saved Adaptive-DH checkpoints to compare context-only adaptation
against full-network fine-tuning. Both start with the same source weights,
calibration-selected context, risk threshold and minibatch stream. This isolates
which parameters may change under a shared optimizer recipe; it does not compare
separately optimized learning-rate schedules.

```bash
python -m experiments.compare_update_capacity banks --output /path/to/capacity-run --device cuda
python -m experiments.compare_update_capacity compare --output /path/to/capacity-run --checkpoints /path/to/original-run --seed 7
```

Repeat the second command for seeds 17 and 29, then summarize:

```bash
python -m experiments.summarize_update_capacity /path/to/capacity-run
```

The original run must contain `adh/seed-N/pretrained.pt`. No source model is
retrained. Use a new output directory: the follow-up generates fresh target
training, calibration and evaluation paths, keeping the financial settings
unchanged. Repeating a comparison command resumes from saved adaptation
milestones with optimizer and random-number state intact.

Both methods pay for selecting their starting context. Report distinct target
paths separately from repeated gradient and calibration evaluations. The full
banks are generated up front, so paths consumed are not generation savings.
The saved zero-update policy is also a no-update control. Wall times from
overlapping runs are diagnostic, not an isolated speed comparison.
