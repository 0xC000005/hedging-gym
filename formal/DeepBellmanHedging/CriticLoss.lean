import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# The critic regression targets (paper footnote 14)

The paper fits the critic by the nested loss `E[(−V(S) + E[h(S') | S] + g(S))²]`
(equation (12)) but implements the unconditional loss `E[(−V(S) + h(S') + g(S))²]`
(equation (13)), arguing that both have the same gradient in `V`. On a finite joint
sample the two losses differ exactly by the mean conditional variance of `h`, which
does not depend on `V`, so they have the same minimisers and gradients.
-/

open Finset

noncomputable section

/-- Finite joint sample of today's state and tomorrow's outcome, every state visited. -/
structure JointLaw (S Ω : Type*) [Fintype S] [Fintype Ω] where
  weight : S → Ω → ℝ
  nonneg : ∀ s ω, 0 ≤ weight s ω
  mass_pos : ∀ s, 0 < ∑ ω, weight s ω

namespace JointLaw

variable {S Ω : Type*} [Fintype S] [Fintype Ω] (P : JointLaw S Ω)

/-- `E[h(S') | S = s]`. -/
def cond (h : S → Ω → ℝ) (s : S) : ℝ := (∑ ω, P.weight s ω * h s ω) / ∑ ω, P.weight s ω

/-- Paper equation (13): `E[(V(S) + h(S') + g(S))²]`. -/
def unconditional (V g : S → ℝ) (h : S → Ω → ℝ) : ℝ :=
  ∑ s, ∑ ω, P.weight s ω * (V s + h s ω + g s) ^ 2

/-- Paper equation (12): `E[(V(S) + E[h(S') | S] + g(S))²]`. -/
def nested (V g : S → ℝ) (h : S → Ω → ℝ) : ℝ :=
  ∑ s, ∑ ω, P.weight s ω * (V s + P.cond h s + g s) ^ 2

/-- `E[Var(h(S') | S)]`, independent of `V`. -/
def conditionalVariance (h : S → Ω → ℝ) : ℝ :=
  ∑ s, ∑ ω, P.weight s ω * (h s ω - P.cond h s) ^ 2

theorem sum_weight_mul_sub_cond (h : S → Ω → ℝ) (s : S) :
    ∑ ω, P.weight s ω * (h s ω - P.cond h s) = 0 := by
  have hm : (∑ ω, P.weight s ω) ≠ 0 := (P.mass_pos s).ne'
  have hc : P.cond h s * ∑ ω, P.weight s ω = ∑ ω, P.weight s ω * h s ω := by
    unfold cond; exact div_mul_cancel₀ _ hm
  calc ∑ ω, P.weight s ω * (h s ω - P.cond h s)
      = ∑ ω, P.weight s ω * h s ω - P.cond h s * ∑ ω, P.weight s ω := by
        rw [mul_sum, ← sum_sub_distrib]
        exact sum_congr rfl fun ω _ ↦ by ring
    _ = 0 := by rw [hc]; ring

/-- The two losses differ by the mean conditional variance, which does not involve `V`. -/
theorem unconditional_eq_nested_add (V g : S → ℝ) (h : S → Ω → ℝ) :
    P.unconditional V g h = P.nested V g h + P.conditionalVariance h := by
  unfold unconditional nested conditionalVariance
  rw [← sum_add_distrib]
  refine sum_congr rfl fun s _ ↦ ?_
  have key := P.sum_weight_mul_sub_cond h s
  have expand : ∀ ω, P.weight s ω * (V s + h s ω + g s) ^ 2
      = P.weight s ω * (V s + P.cond h s + g s) ^ 2 + P.weight s ω * (h s ω - P.cond h s) ^ 2
        + 2 * (V s + P.cond h s + g s) * (P.weight s ω * (h s ω - P.cond h s)) :=
    fun ω ↦ by ring
  simp only [expand, sum_add_distrib, ← mul_sum, key, mul_zero, add_zero]

/-- Consequently both losses have the same differences in `V`, hence the same minimisers. -/
theorem unconditional_sub_eq (V W g : S → ℝ) (h : S → Ω → ℝ) :
    P.unconditional V g h - P.unconditional W g h = P.nested V g h - P.nested W g h := by
  rw [P.unconditional_eq_nested_add, P.unconditional_eq_nested_add]
  ring

end JointLaw

end
