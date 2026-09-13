# Formal proofs for Deep Bellman Hedging

Lean 4 / Mathlib formalisation of the theorems in Buehler, Murray and Wood,
[Deep Bellman Hedging](https://arxiv.org/abs/2207.00932v4) (revision 2.03, April
2024), the paper behind `hedging_gym.baselines.deep_bellman_hedging`. This
directory is a self-contained Lake project; it is not part of the Python package
or its test suite.

## Build

```bash
cd formal
lake exe cache get   # prebuilt Mathlib, once
lake build
```

`lean-toolchain` pins the Lean release and `lake-manifest.json` pins Mathlib.
There is no `sorry` in the project.

## Theorem map

| Paper | Lean |
|---|---|
| §2.1 monetary utility: monotone, cash-invariant | `MonetaryUtility` (`MonetaryUtility.lean`) |
| §5 proof of Theorem 1, contraction step and Banach fixed point | `contractingWith_of_monotone_cash`, `existsUnique_fixedPoint_of_monotone_cash`, `tendsto_iterate_of_monotone_cash` (`Contraction.lean`) |
| Definition 1, equation (3), Bellman operator `T` | `HedgingModel.bellman` (`Bellman.lean`) |
| Theorem 1 (`th:convergence`): unique finite solution | `HedgingModel.existsUnique_value`; value iteration `HedgingModel.tendsto_iterate_value` |
| Equation (18), vanilla Deep Hedging operator `T̃`; §5.1 discounted cash-invariance | `VanillaModel.bellman`, `VanillaModel.bellman_cash` (`VanillaBellman.lean`) |
| Theorem 3 (`th:convergence_dh`), existence and uniqueness | `VanillaModel.existsUnique_value` |
| Finite maturity with zero rates, the setting the Python baseline implements | `existsUnique_fixedPoint_of_horizon`, `HedgingModel.existsUnique_value_of_horizon` (`FiniteHorizon.lean`) |
| Definition 2, optimized certainty equivalent is a monetary utility | `FiniteLaw.toMonetaryUtility` (`OCE.lean`); entropic and CVaR utilities `entropicUtility_le`, `cvarUtility_le` |
| Footnote 14, unconditional versus nested critic loss | `JointLaw.unconditional_eq_nested_add`, `JointLaw.unconditional_sub_eq` (`CriticLoss.lean`) |
| §4 finite statistical arbitrage bound (footnote 16) | `utility_gains_le_of_finite_arbitrage` (`StatArb.lean`) |

## Modelling choices and gaps

- A monetary utility is an abstract operator on functions of tomorrow's outcome,
  indexed by today's state. Only monotonicity and cash-invariance are used, as the
  paper notes after Theorem 1; measurability, normalisation and concavity are not
  modelled for the fixed-point results. `OCE.lean` works on a finite sample space
  with probability weights, matching the paper's finite training sample.
- Value functions live in the bounded functions `S →ᵇ ℝ` with the supremum norm,
  the case the paper proves; `S` carries the discrete topology so that every
  function is continuous. The unbounded extension the paper cites is not covered.
- The paper's "rewards are finite" hypothesis (equation (4)) is pointwise in the
  state. The sup-norm argument needs `T0` to be bounded, so `HedgingModel.reward_bound`
  is uniform over states and admissible actions. This is a strengthening of the
  stated hypothesis that the paper's proof uses implicitly.
- Admissible action sets are nonempty subsets of a fixed action type; infinite
  transaction costs are represented by exclusion from the admissible set.
- `VanillaModel` assumes normalised utilities, `U_s(0) = 0`, to bound the discounted
  continuation value; the paper's monetary utilities are normalised by definition.
- Theorem 2, that for a time-consistent utility (entropic or expectation) the vanilla
  Deep Hedging value function satisfies the Bellman equation, is not formalised; only
  Theorem 3's existence and uniqueness is.
- The finite-horizon result adds what the paper leaves implicit for fixed
  maturities: with `β = 1` and the continuation masked from the horizon on, the
  operator ignores its input after `N + 1` steps, so backward induction gives the
  unique solution without a contraction.
- `StatArb.lean` states the proposition with the footnote's ingredients (additive
  expectation, risk aversion `U ≤ E`, and additivity of gains in the portfolio) as
  hypotheses, rather than constructing the infinite-horizon gains process.
- The multi-step operator `T_n` of the paper's Remark 2, and its coincidence with the
  iterate `T^[n]` for the entropic utility at `β = 1` (with discounting the two differ
  even then, since a discount factor inside the utility rescales the risk aversion), are
  not formalised: `HedgingModel` carries one utility per state and no product structure
  on outcomes.
