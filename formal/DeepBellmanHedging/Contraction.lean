import Mathlib.Topology.ContinuousMap.Bounded.Normed
import Mathlib.Topology.MetricSpace.Contracting

/-!
# Monotone, cash-subinvariant operators are contractions

The core of both existence theorems in *Deep Bellman Hedging* (section 5, proof of
Theorem 1): an operator on bounded functions that is monotone and shifts by at most
`β·c` when its argument shifts by a constant `c ≥ 0` is a `β`-contraction in the
supremum norm. The Banach fixed-point theorem then gives a unique fixed point and
convergence of value iteration, which is the paper's footnote 18 argument.
-/

open BoundedContinuousFunction Filter

variable {S : Type*} [TopologicalSpace S]

/-- Paper section 5: `Tf ≤ T(g + ‖f - g‖) ≤ Tg + β‖f - g‖`, and symmetrically. -/
theorem contractingWith_of_monotone_cash {T : (S →ᵇ ℝ) → (S →ᵇ ℝ)} {β : NNReal} (hβ : β < 1)
    (hmono : Monotone T)
    (hcash : ∀ (f : S →ᵇ ℝ) (c : ℝ), 0 ≤ c → T (f + const S c) ≤ T f + const S (β * c)) :
    ContractingWith β T := by
  have key : ∀ f g : S →ᵇ ℝ, ∀ s, T f s ≤ T g s + β * dist f g := by
    intro f g s
    have hfg : f ≤ g + const S (dist f g) := fun x ↦ by
      change f x ≤ g x + dist f g
      have hx := dist_coe_le_dist (f := f) (g := g) x
      rw [Real.dist_eq] at hx
      linarith [le_abs_self (f x - g x)]
    have h := (hmono hfg).trans (hcash g (dist f g) dist_nonneg) s
    change T f s ≤ T g s + β * dist f g at h
    exact h
  refine ⟨hβ, LipschitzWith.of_dist_le_mul fun f g ↦ ?_⟩
  refine (dist_le (mul_nonneg (NNReal.coe_nonneg β) dist_nonneg)).mpr fun s ↦ ?_
  have k₁ := key f g s
  have k₂ := key g f s
  rw [dist_comm g f] at k₂
  rw [Real.dist_eq, abs_sub_le_iff]
  constructor <;> linarith

/-- Unique fixed point of a monotone, cash-subinvariant operator with `β < 1`. -/
theorem existsUnique_fixedPoint_of_monotone_cash {T : (S →ᵇ ℝ) → (S →ᵇ ℝ)} {β : NNReal}
    (hβ : β < 1) (hmono : Monotone T)
    (hcash : ∀ (f : S →ᵇ ℝ) (c : ℝ), 0 ≤ c → T (f + const S c) ≤ T f + const S (β * c)) :
    ∃! V : S →ᵇ ℝ, T V = V := by
  have h := contractingWith_of_monotone_cash hβ hmono hcash
  exact ⟨h.fixedPoint, h.fixedPoint_isFixedPt,
    fun W hW ↦ h.fixedPoint_unique' hW h.fixedPoint_isFixedPt⟩

/-- Value iteration `V⁽ⁿ⁾ = T V⁽ⁿ⁻¹⁾` converges to the fixed point from any start
(footnote 18). -/
theorem tendsto_iterate_of_monotone_cash {T : (S →ᵇ ℝ) → (S →ᵇ ℝ)} {β : NNReal}
    (hβ : β < 1) (hmono : Monotone T)
    (hcash : ∀ (f : S →ᵇ ℝ) (c : ℝ), 0 ≤ c → T (f + const S c) ≤ T f + const S (β * c))
    (V₀ : S →ᵇ ℝ) :
    ∃ V : S →ᵇ ℝ, T V = V ∧ Tendsto (fun n ↦ T^[n] V₀) atTop (nhds V) := by
  have h := contractingWith_of_monotone_cash hβ hmono hcash
  exact ⟨h.fixedPoint, h.fixedPoint_isFixedPt, h.tendsto_iterate_fixedPoint V₀⟩
