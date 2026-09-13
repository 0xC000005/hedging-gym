import Mathlib.Order.ConditionallyCompleteLattice.Basic
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Tactic

/-!
# Finite statistical arbitrage bounds the risk-adjusted gains (paper section 4)

Paper proposition: if the market has only finite statistical arbitrage,
`γ = sup_π E[G^π(0)] < ∞`, then `U[G^π(z)] ≤ E[G^0(z)] + γ` for every policy `π` and
portfolio `z`. The footnote proof uses only risk aversion `U ≤ E`, additivity of the
expectation, and the decomposition `G^π(z) = G^0(z) + G^π(0)` of gains into the
untraded portfolio's gains and the gains of trading from an empty book; those are the
hypotheses here.
-/

variable {Policy Portfolio Ω : Type*}

/-- Paper section 4 proposition, with the footnote's ingredients as hypotheses. -/
theorem utility_gains_le_of_finite_arbitrage
    (G : Policy → Portfolio → Ω → ℝ) (E U : (Ω → ℝ) → ℝ)
    (hE : ∀ X Y, E (fun ω ↦ X ω + Y ω) = E X + E Y)
    (hU : ∀ X, U X ≤ E X)
    (π₀ : Policy) (z₀ : Portfolio)
    (hG : ∀ π z ω, G π z ω = G π₀ z ω + G π z₀ ω)
    (hγ : BddAbove (Set.range fun π ↦ E (G π z₀)))
    (π : Policy) (z : Portfolio) :
    U (G π z) ≤ E (G π₀ z) + ⨆ π', E (G π' z₀) := by
  have hsplit : E (G π z) = E (G π₀ z) + E (G π z₀) := by
    rw [← hE]
    congr 1
    funext ω
    exact hG π z ω
  calc U (G π z) ≤ E (G π z) := hU _
    _ = E (G π₀ z) + E (G π z₀) := hsplit
    _ ≤ E (G π₀ z) + ⨆ π', E (G π' z₀) := add_le_add le_rfl (le_ciSup hγ π)
