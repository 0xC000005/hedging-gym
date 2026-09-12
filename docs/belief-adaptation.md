# Belief-FB encoder transfer into Deep Hedging

This lane tests a **Belief-FB-inspired dynamics-context encoder plus Deep
Hedging**, not full Belief-FB, Rotation-FB, a reproduction, or a claimed superior
hedger. The direct terminal expected-shortfall objective and complete portfolio
cash ledger are unchanged. The common observation still includes the declared
Heston parameters. Consequently this tests whether learned transition context
adds useful conditioning alongside known parameters, not hidden-regime recovery.

## Pinned source and actual reuse

The author repository is [maxsbob/BeliefConditionedFB](https://github.com/maxsbob/BeliefConditionedFB),
pinned at `30e7487ca033c3619ec744ed55f916ece005c425`. The external checkout used for
inspection is under the run directory's `donors/belief-fb`, outside Git.
The paper is [Zero-Shot Adaptation of Behavioral Foundation Models to Unseen
Dynamics, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cd7fe6d521c3fc74c5363a7e29f47910-Abstract-Conference.html).

The implementation is a modified PyTorch port of these native artifacts:

- [`utils/transformer_nets.py`](https://github.com/maxsbob/BeliefConditionedFB/blob/30e7487ca033c3619ec744ed55f916ece005c425/utils/transformer_nets.py):
  `DynamicsTransformer`, `Encoder1DBlock`, `MlpBlock`, and `NextStatePrediction`.
- [`agents/dynamics_fb.py`](https://github.com/maxsbob/BeliefConditionedFB/blob/30e7487ca033c3619ec744ed55f916ece005c425/agents/dynamics_fb.py):
  `context_encoder_loss`.
- The upstream Apache-2.0 license is retained in [belief-fb-LICENSE.txt](belief-fb-LICENSE.txt).

Recognizable source mechanisms retained: separate state/action/next-state
projections; concatenated transition tokens; no positional embeddings; unmasked
pre-normalized self-attention and GELU residual blocks; mean pooling across
transitions; Gaussian context mean and log standard deviation; a reparameterized
context sample used by the next-state predictor; and mean half squared error.
The inspected released loss does **not** add KL regularization. We do not infer
such an objective from the paper's Gaussian-prior description.

The port changes Flax/JAX to the existing PyTorch stack and uses smaller declared
dimensions, PyTorch initializers, zero dropout, Adam, and gradient clipping.
For financial units, inputs are `(log(S/S0)/0.2, variance/0.04)`; 0.04 is fixed,
not a normalization by current theta or v0. The action channel is zero because
trading does not alter this benchmark's market process. No finance dynamics
equations are copied or changed. The predictor reconstructs the same normalized
next states supplied in the context, as in the source loss; it is not a
held-out probabilistic forecasting loss.

Full-source execution is not claimed. In this pinned checkout,
`main_dynamics_continuous.py` has `else: pass` for the context-enabled training
branch, so that entry point does not provide a complete runnable continuous
training experiment as written. Native JAX/Flax execution is not part of the
PyTorch integration qualification.

## Why the full algorithm is not transplanted

The native forward-backward factorization extracts policies for expected
discounted state rewards from a successor measure. Our objective is a
cost-inclusive terminal loss distribution with a learned ES threshold. Treating
mean per-step rewards as this tail objective would change the financial task.
An augmented-state and risk-aware derivation would be additional algorithm work;
this lane does not claim it was done. It transfers the pretrained context
inference mechanism into the existing directly differentiated hedger instead.

The donor's original evaluation keeps its hidden dynamics context fixed within
an episode and requires a target-environment transition history. It does not
establish tracking of changes within an episode, identification from tiny
financial samples, or improved ES. In particular, its conditional-mean objective
may ignore parameters that mainly change variance/covariance, such as vol-of-vol
or correlation. No convenient auxiliary volatility or parameter-label loss has
been added. The first comparison should report this limitation if contexts fail
to distinguish regimes or fail to improve risk.

## Common-trainer API

```python
from functools import partial
import torch
from hedging_gym.baselines.belief_adaptation import (
    BeliefEmbeddedPolicy, encode_bank_context, train_belief_encoder,
)

encoder, encoder_work = train_belief_encoder(
    source_history_banks, seed=seed, updates=1000, batch_size=32,
    context_length=20, embedding_dim=4, device=device,
)
encoded = [encode_bank_context(encoder, bank, history_paths=32,
                               context_length=20)
           for bank in source_history_banks]
source_contexts = torch.stack([value for value, work in encoded])
policy_factory = partial(BeliefEmbeddedPolicy, encoder=encoder,
                         source_contexts=source_contexts)
# Pass policy_factory(config, n_tasks, embedding_dim, hidden) to the common
# multitask trainer. n_tasks/embedding_dim/hidden are keyword arguments.
```

During common financial pretraining, `active_task` selects the corresponding
fixed source context. The optimizer may include `source_embeddings`, but they
remain frozen and have no gradients. The encoder is frozen and is excluded from
the financial optimizer; only shared policy weights and the common source ES
thresholds learn. This is a disclosed difference from AdaptiveDH, which jointly
learns its source task vectors. Shared geometry, observations, book, horizon,
paths, and terminal objective remain common.

Before target evaluation, call:

```python
history_work = policy.infer_context(
    target_prior_history_bank, history_paths=32, context_length=20,
)
# Evaluate this fixed inferred vector before any target ES updates.
# The existing AdaptationUpdater can then fit the embedding on the separate
# target training bank; reset_embedding() retains the freshly inferred vector.
```

Inference averages posterior means from the first 20 transitions of 32 declared
completed prior paths. It does not read option marks/payoffs or change weights.
There is no hidden evaluation-market vector lookup. Evaluation never infers a
context from the path currently being hedged. The caller must supply the fresh
prior bank for every new target stage, including a return to a previously seen
market; the adapter does not silently restore previous calibrated embeddings.

## Information and compute contract

The common harness owns bank seeds and disjointness. Source history banks must
be distinct from the financial training/evaluation banks; the target prior bank
must be independently generated before target evaluation and distinct from the
ES-update and evaluation banks. The adapter cannot establish that contract from
an unlabelled tensor alone and does not pretend to do so with a filename check.

The encoder records all source paths/transitions available, training transition
presentations (`updates * batch_size * context_length`), parameter count, and
wall-clock training time. Every inference records paths/transitions consumed,
prior-bank size, inference time, context norm, and variation of posterior means
across the supplied prior paths. The harness must additionally count generation
of the full prior banks and all encoder work in total method cost. No zero-shot
label should conceal its required history or that pretraining cost.

For example, a 1,024-path prior bank with a full N-step book generates 1,024*N
transitions even when inference consumes only 32*20. That generation remains a
cost. The same prior observations must be available to competing methods if any
method uses them for adaptation; they do not replace the declared target ES
training budget.

Useful diagnoses include source-context separation relative to variation across
prior paths, stability over fresh prior histories, and the actual terminal-ES
curve versus the parameter-observing AdaptiveDH baseline. Good reconstruction
loss alone does not demonstrate identification or hedging value. Known-parameter
conditioning substantially weakens the donor's original motivation here, and a
negative result is informative for this particular transfer.

## Qualification

Focused tests cover transition permutation invariance, nonzero gradients in the
native Gaussian encoder/predictor mechanism, reproducible training, invariance
to unconsumed history and payoff/mark changes, preservation of observed Heston
parameters, frozen source vectors/encoder, complete-book terminal-ES gradients,
latent-only adaptation, unchanged state during evaluation, and saved prior
context restoration. Tests use a small three-step complete book for API and
gradient qualification; the comparative experiment keeps the common full
horizon. They do not establish financial superiority or numerical Flax parity.
