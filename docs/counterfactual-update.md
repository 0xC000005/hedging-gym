# Counterfactual terminal-risk updates

This research prototype asks a narrow question: can a hybrid hedger learn its
HOLD/TRADE choices better by evaluating every available mode, rather than only
the sampled choice? It reuses `HybridPolicy`, the market bank and the cash ledger.
Continuous sizing and the value network remain frozen.

At one uniformly sampled trading date, the current policy generates a causal
history. Every mode then receives its own rollout to expiry. Branches share
market shocks and future random uniforms, but their future trades depend on
their own holdings and observations. Each branch is scored using its own full
terminal loss and the same fixed training-calibrated ES threshold.

The update differentiates the probability-weighted sum of those terminal costs.
The comparison arm samples one mode and gets more independent histories to
approximately match decision-ledger work. Both recollect on-policy data after
each update. This is a categorical-only experiment, not full HPO.

## Where the idea comes from

- [Expected policy gradients](https://jmlr.org/papers/v21/18-012.html): integrate
  action credit instead of sampling just one action.
- [Input-dependent baselines](https://openreview.net/forum?id=Hyg1G2AqtQ) and
  [counterfactual credit assignment](https://proceedings.mlr.press/v139/mesnard21a.html):
  separate external uncertainty from action quality without leaking future
  information into the deployed policy.
- [HPO](https://arxiv.org/abs/2605.14297): the source hybrid policy architecture;
  this pilot does not replace its full mixed-gradient algorithm.

All-mode averaging and its variance argument are established ideas. This
prototype does not claim a new theorem, globally optimal policy or superiority
over published hedging methods.

## Run and inspect

Use a qualification directory containing the saved training/development banks
and HPO stage snapshots. New outputs must go elsewhere:

```bash
uv run --frozen python -m benchmarks.qualify_counterfactual \
  --source-dir /path/to/qualification \
  --output /path/to/new-pilot \
  --preset operational_fixed --seeds 7 17 29 \
  --updates 100 --batch-size 64 --device cuda

uv run --frozen python -m benchmarks.review_counterfactual \
  --runs /path/to/new-pilot --output /path/to/new-review.json

uv run --frozen pytest -q tests/test_counterfactual.py
```

The first command saves progress, checkpoints, training costs and development
trade tapes. The second reconstructs losses using independent NumPy accounting
and checks reported ES. Sampled deployment is primary; greedy deployment is
reported separately. Decision-ledger counts, liquidation counts and elapsed time
are distinct costs, not interchangeable definitions of matched compute.

Three focused tests check action-specific tail costs, branch cash accounting,
and the categorical gradient against exact two-date enumeration. Development
results guide the next experiment; publication claims still require fresh
evaluation paths, strong controls and training-seed uncertainty.
