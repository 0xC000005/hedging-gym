# Learned context retrieval for hedging

The [retrieval extension](../src/hedging_gym/extensions/skill_retrieval.py)
ranks a frozen library of source task vectors for a new market, then passes the
selected vector to the existing embedding-adaptation trainer. It changes the
initialization, not the financial environment or update objective.

## Source and mapping

[SRSA, ICLR 2025](https://arxiv.org/abs/2503.04538) retrieves specialist policies
using predicted transfer success. The
[official implementation](https://github.com/NVlabs/SRSA/tree/2bed3f7ecee73be29eaeccd1b3a6fe03d4482702)
is pinned at `2bed3f7ecee73be29eaeccd1b3a6fe03d4482702`.
Relevant files are `source/SRSA/SRSA/embedding/{dataset,model,train}.py`:
`PairData` constructs ordered task pairs, `SuccPreNetwork.fc` supplies the
one-hidden-layer ReLU/MSE predictor, and `retrieve` ranks skills.
Xavier weights and 0.01 biases are retained.

| SRSA component | Finance adaptation |
|---|---|
| Separate specialist policies | Shared hedger with frozen source contexts |
| Geometry, dynamics and expert-action features | Observed market parameters |
| Higher zero-shot success | Lower cost-inclusive terminal ES |
| 128-unit predictor | Configurable head, default width 32 |
| PPO and self-imitation | Existing embedding-only gradient updates |

The robot encoders, full-policy library, self-imitation and expanding library
are not implemented. This is a transfer of the retrieval mechanism, not a
complete SRSA reproduction.

## Labels and supported inputs

For source context i on source market j, fit the target

```text
label[i, j] = ES(context_i, market_j) - ES(mean_context, market_j)
```

Centering removes a market-specific offset without changing rankings. Feature
normalization and label scaling use source markets only. All ordered pairs are
used; source and target order matters.

Provide at least two ordered source contexts and matching fresh calibration
banks. Banks must share book, calendar, execution and risk settings. Both
market model and discretization scheme must agree across the source library
and target; categorical model/scheme identifiers are excluded from numerical
features. The supplied comparison uses ES95.

Scoring pools losses before computing ES and preserves the source policy.
Predictor fitting and top-one ranking do not use target evaluation paths.
Top-five or exhaustive rescoring requires a separate target-calibration bank
and additional reported work.

## API

Given a pretrained `source_policy`, separate `source_calibration_banks` and
`target_training_bank`, the public adaptation factory accepts the retrieved
initialization explicitly:

```python
from copy import deepcopy
from hedging_gym.baselines.adaptive_deep_hedging import make_updater
from hedging_gym.extensions.skill_retrieval import rank_contexts, train_retriever

retriever, retrieval_work = train_retriever(
    source_policy, source_calibration_banks, seed=7, updates=300,
)
ranking = rank_contexts(retriever, source_policy, target_training_bank.config)
initial = source_policy.source_embeddings[ranking[0]].detach().clone()
target_policy = deepcopy(source_policy)
updater = make_updater(target_policy, seed=7, updates=50)
updater(target_training_bank, initial_embedding=initial)
```

Passing `initial_embedding` at the first new-market update prevents the default
mean-context reset from erasing the selection. Use identical update budgets,
learning rates and reset rules for mean, nearest and learned retrieval controls.
The underlying continuous adaptation requires finite bounds and rejects
minimum-order sizes and trade lots.

## Run and verify

The Adaptive-DH arm of the common comparison also runs nearest, top-one,
top-five and exhaustive retrieval:

```bash
uv run --frozen python -m benchmarks.compare_fast_adaptation banks \
  --output /path/to/fast-adaptation --device cpu
uv run --frozen python -m benchmarks.compare_fast_adaptation compare \
  --output /path/to/fast-adaptation --method adh --seed 7 --device cpu
uv run --frozen pytest -q tests/test_skill_retrieval.py
```

Use an external output directory; add `--smoke` to both comparison stages in a
separate directory for a small execution check. See
[fast adaptation](fast-adaptation.md) for multiple seeds and summaries.

Report source pretraining, calibration generation, all label rollouts,
predictor fitting, rescoring and adaptation costs. Outputs retain scores,
labels, normalizations, timing and checkpoint state. Good source-label fit
alone does not establish useful ranking on unseen markets or faster adaptation.
