# Belief-FB encoder transfer into Deep Hedging

[BeliefEmbeddedPolicy](../src/hedging_gym/baselines/belief_adaptation.py)
conditions Deep Hedging on a context inferred from completed market histories.
The encoder is pretrained separately, then frozen during financial training.
Declared market parameters remain observable. This tests additional learned
conditioning; it does not identify a hidden regime or implement full Belief-FB
or Rotation-FB.

## Sources and license

The source is [Zero-Shot Adaptation of Behavioral Foundation Models to Unseen
Dynamics, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cd7fe6d521c3fc74c5363a7e29f47910-Abstract-Conference.html)
and [BeliefConditionedFB](https://github.com/maxsbob/BeliefConditionedFB/tree/30e7487ca033c3619ec744ed55f916ece005c425),
pinned at `30e7487ca033c3619ec744ed55f916ece005c425`.

This modified PyTorch port transfers:

- `DynamicsTransformer`, `Encoder1DBlock`, `MlpBlock` and
  `NextStatePrediction` from
  [utils/transformer_nets.py](https://github.com/maxsbob/BeliefConditionedFB/blob/30e7487ca033c3619ec744ed55f916ece005c425/utils/transformer_nets.py).
- `context_encoder_loss` from
  [agents/dynamics_fb.py](https://github.com/maxsbob/BeliefConditionedFB/blob/30e7487ca033c3619ec744ed55f916ece005c425/agents/dynamics_fb.py).

The upstream Apache-2.0 license is retained in
[belief-fb-LICENSE.txt](belief-fb-LICENSE.txt). Preserve this notice when
redistributing the adapted code.

## Mechanism and deviations

The encoder retains separate state/action/next-state projections, concatenated
transition tokens, no positional embeddings, unmasked pre-normalized attention,
GELU residual blocks, and mean pooling across transitions. It predicts a
Gaussian context; a reparameterized sample feeds the next-state predictor.
Training uses mean half squared error, with no KL term, matching the released
source loss.

The port replaces Flax/JAX with PyTorch, smaller declared dimensions, PyTorch
initializers, zero dropout, Adam and gradient clipping. Financial states are
`(log(S/S0)/0.2, variance/0.04)`, with fixed normalization constants. The action
channel is zero because hedge trades do not affect the market process. The
predictor reconstructs the normalized next states present in its input context;
its loss is not a held-out forecasting score.

Financial pretraining learns shared hedge-policy weights and source ES
thresholds. Source context vectors and the encoder remain frozen, unlike
Adaptive DH's jointly learned source vectors. At a target market, posterior
means from completed prior histories initialize the context; optional terminal-ES
updates then adjust that context through the existing adaptation interface.

Full forward-backward policy learning is not transferred: expected discounted
state rewards do not directly represent the cost-inclusive terminal tail-risk
objective. Native JAX/Flax execution and numerical cross-framework parity are
not established. The pinned `main_dynamics_continuous.py` entry point leaves
its context-enabled training branch unimplemented.

## Inputs and supported configuration

`train_belief_encoder` requires at least two nonempty history banks with the
same book, calendar, execution and risk settings. Each must contain at least
`context_length` transitions. `encode_bank_context` defaults to the first
20 transitions of 32 completed paths and averages posterior means.
`BeliefEmbeddedPolicy` inherits finite holding bounds and continuous adaptation;
minimum-order sizes and trade lots are unsupported.

Supply separate banks for source history, financial training, target prior
history, target adaptation and evaluation. The adapter cannot infer independence
from tensor contents. Before every target stage, call
`policy.infer_context(target_prior_history_bank, ...)`, including when returning
to a previously visited market. It never infers context from the evaluation
path currently being hedged or silently restores a prior target vector.
`reset_embedding()` preserves the freshly inferred initialization.

## Run

The common comparison constructs the encoder, source contexts and hedger:

```bash
uv run --frozen python -m benchmarks.compare_fast_adaptation banks \
  --output /path/to/fast-adaptation --device cpu
uv run --frozen python -m benchmarks.compare_fast_adaptation compare \
  --output /path/to/fast-adaptation --method belief --seed 7 --device cpu
```

Choose an external output directory. Add `--smoke` to both commands in a
separate directory for an execution check. See
[fast adaptation](fast-adaptation.md) for the full comparison and controls.

```bash
uv run --frozen pytest -q tests/test_belief_adaptation.py
```

Tests cover encoder gradients, transition-order invariance, consumed-history
boundaries, frozen financial pretraining components, context updates and shared
cash accounting. They do not establish improved financial performance.

## Interpretation and cost

Charge encoder pretraining, generation of every prior bank, context inference
and target updates. Consuming 32 paths from a larger generated bank does not
erase the cost of generating that bank. Give competing adaptation methods the
same prior information when comparing them.

A next-state mean-reconstruction objective may poorly distinguish parameters
that mainly affect variance or covariance. Reconstruction fit alone therefore
does not establish dynamics identification or useful hedging context. The
encoder assumes a fixed market within each supplied history; within-episode
regime tracking is not demonstrated.
