# Learned skill retrieval for hedging adaptation

When a market changes, which previously learned hedging context is the best
starting point? This adapter learns that choice, then leaves adaptation to the
existing direct-gradient expected-shortfall trainer. It does not change the
market simulator, financial book, transaction costs or accounting.

## Source and finance mapping

[SRSA (Guo et al., ICLR 2025)](https://arxiv.org/abs/2503.04538) evaluates each
source policy on each prior task and learns to predict zero-shot transfer
success. It retrieves a promising source policy and fine-tunes it. The
[official code](https://github.com/NVlabs/SRSA) is pinned at
`2bed3f7ecee73be29eaeccd1b3a6fe03d4482702`.

The relevant source files are
`source/SRSA/SRSA/embedding/{dataset,model,train}.py`: `PairData` forms ordered
source–target pairs, `SuccPreNetwork.fc` is a one-hidden-layer ReLU scalar
predictor trained with squared error, and `retrieve` ranks source skills.
The head's Xavier initialization and 0.01 biases are retained. We implement
this small head directly; loading its enclosing network would also require
unneeded point-cloud and robot-transition encoders.

| SRSA | This finance adapter |
| --- | --- |
| Separate specialist policies | Fixed shared hedger plus source task vectors |
| Geometry, dynamics and expert-action features | Supplied stochastic-market parameters |
| Higher zero-shot success | Lower cost-inclusive terminal ES95 |
| 128-unit predictor over 90 skills | 32-unit predictor over the initial 8 contexts |
| PPO and self-imitation after retrieval | Existing embedding-only ES gradient updates |

This is **SRSA-style learned retrieval**, not a native reproduction or complete
SRSA implementation. The donor's learned representation, full-policy library,
self-imitation and expanding library are not implemented. In particular, the
original hypothesis that good zero-shot transfer predicts fast fine-tuning
remains an empirical question for this context-restricted finance adaptation.

## Labels and selection

Let `R[i,j]` be the pooled ES95 of source context `i` on fresh calibration paths
from source market `j`. Let `R[mean,j]` be the result with the mean source
context. The predictor learns

```
y[i,j] = R[i,j] - R[mean,j]
```

Centering on the mean context removes a target-market difficulty offset without
changing source rankings. A single source-calibrated root-mean-square label
scale conditions the MSE optimization. Input means and standard deviations
also come only from source markets; constant source coordinates use unit scale.
All ordered source–target pairs are used, with asymmetric concatenated inputs.

Calibration banks must have fresh, recorded seeds and be separate from source
policy training, future adaptation training and evaluation banks. Chunking is
only for memory: losses are pooled before ES, not averaged after computing ES
on individual chunks. The source policy is copied for scoring and remains
unchanged. No future target paths enter predictor fitting or ranking.

```python
retriever, metadata = train_retriever(
    source_policy, source_calibration_banks,
    seed=7, updates=300, checkpoint_path=artifact_dir / "retriever.pt",
)
ranking = rank_contexts(retriever, source_policy, target_config)
context = source_policy.source_embeddings[ranking[0]].detach().clone()
```

The runner must install this initialization at the new-market adaptation reset;
otherwise `AdaptationUpdater`'s default mean reset erases it. Learning rates,
threshold initialization, number of updates and optimizer reset rules must be
the same for mean, nearest and learned retrieval.

The source paper also evaluates a top-five shortlist on the new task before
choosing one skill. `rank_contexts` supports this source-backed variant: the
runner rescores its first five contexts on a dedicated target-calibration bank,
not an evaluation bank. It must charge that work and disclose the difference
from fast prediction-only top-one selection. With only eight source contexts,
exhaustively rescoring all eight is a useful cheap control, not an oracle for
future realized paths.

## What the comparison must report

- Mean-source initialization, nearest standardized market and learned retrieval,
  before and after identical embedding-update budgets.
- Source pretraining, calibration generation, all `(n+1) × n` label evaluations,
  predictor fitting, any top-k rescoring and target adaptation costs separately.
- Held-out target ES and context selection; label fit is not evidence of
  adaptation gain. Good source fit but poor new-market ranking indicates a
  retrieval generalization problem, not a failure of AdaptiveDH or SRSA.

The adapter returns raw source transfer scores, centered labels, fitted scores,
timings, episode counts and ledger-decision counts. Checkpoints preserve the
predictor, normalization, optimizer, RNG and labels outside Git. Market bank
generation and shared pretraining belong to the common runner's cost ledger.

## Focused verification

`tests/test_skill_retrieval.py` checks pooled label calculations against complete
rollouts, preservation of the source policy, the ReLU/MSE head and gradients,
selection direction, nearest-market control, and checkpointed predictions.
These are implementation checks, not a finance performance claim.
