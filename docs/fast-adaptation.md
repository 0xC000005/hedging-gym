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
