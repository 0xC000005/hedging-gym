import DeepBellmanHedging.MonetaryUtility
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Basic

/-!
# Optimized certainty equivalents are monetary utilities

Paper definition 2: `U[X] = sup_y E[u(X + y)] - y` for a monotone utility `u` with
`u(0) = 0`, `u'(0) = 1`. Normalisation and concavity imply `u(x) ≤ x`, which is the only
consequence needed here; it is taken as a hypothesis. The sample space is finite with
probability weights, matching the paper's finite training sample `Q` (section 3), so
expectations are finite sums. The paper's entropic and CVaR utilities are shown to
satisfy the hypotheses.
-/

open Finset

/-- Probability weights on a finite sample space. -/
structure FiniteLaw (Ω : Type*) [Fintype Ω] where
  weight : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ weight ω
  total : ∑ ω, weight ω = 1

namespace FiniteLaw

variable {Ω : Type*} [Fintype Ω] (P : FiniteLaw Ω)

/-- `E[X]` under the weights. -/
def expect (X : Ω → ℝ) : ℝ := ∑ ω, P.weight ω * X ω

theorem expect_mono {X Y : Ω → ℝ} (h : ∀ ω, X ω ≤ Y ω) : P.expect X ≤ P.expect Y :=
  sum_le_sum fun ω _ ↦ mul_le_mul_of_nonneg_left (h ω) (P.nonneg ω)

theorem expect_add_const (X : Ω → ℝ) (c : ℝ) : P.expect (fun ω ↦ X ω + c) = P.expect X + c := by
  unfold expect
  simp only [mul_add, sum_add_distrib, ← sum_mul, P.total, one_mul]

/-- Paper definition 2: `U[X] = sup_y E[u(X + y)] - y`. -/
noncomputable def oce (u : ℝ → ℝ) (X : Ω → ℝ) : ℝ :=
  ⨆ y : ℝ, P.expect (fun ω ↦ u (X ω + y)) - y

variable {u : ℝ → ℝ}

theorem oce_term_le (hux : ∀ x, u x ≤ x) (X : Ω → ℝ) (y : ℝ) :
    P.expect (fun ω ↦ u (X ω + y)) - y ≤ P.expect X := by
  have := P.expect_mono (X := fun ω ↦ u (X ω + y)) (Y := fun ω ↦ X ω + y) fun ω ↦ hux _
  rw [P.expect_add_const] at this
  linarith

theorem bddAbove_oce (hux : ∀ x, u x ≤ x) (X : Ω → ℝ) :
    BddAbove (Set.range fun y : ℝ ↦ P.expect (fun ω ↦ u (X ω + y)) - y) :=
  ⟨P.expect X, by rintro _ ⟨y, rfl⟩; exact P.oce_term_le hux X y⟩

theorem oce_mono (hu : Monotone u) (hux : ∀ x, u x ≤ x) : Monotone (P.oce u) := by
  intro X Y hXY
  refine ciSup_mono (P.bddAbove_oce hux Y) fun y ↦ ?_
  have : P.expect (fun ω ↦ u (X ω + y)) ≤ P.expect (fun ω ↦ u (Y ω + y)) :=
    P.expect_mono fun ω ↦ hu (by linarith [hXY ω])
  linarith

theorem oce_cash (hux : ∀ x, u x ≤ x) (X : Ω → ℝ) (c : ℝ) :
    P.oce u (fun ω ↦ X ω + c) = P.oce u X + c := by
  apply le_antisymm
  · refine ciSup_le fun y ↦ ?_
    change P.expect (fun ω ↦ u (X ω + c + y)) - y ≤ P.oce u X + c
    have h : P.expect (fun ω ↦ u (X ω + (y + c))) - (y + c) ≤ P.oce u X :=
      le_ciSup (P.bddAbove_oce hux X) (y + c)
    have e : P.expect (fun ω ↦ u (X ω + c + y)) = P.expect (fun ω ↦ u (X ω + (y + c))) := by
      congr 1; funext ω; congr 1; ring
    rw [e]
    linarith
  · have : P.oce u X ≤ P.oce u (fun ω ↦ X ω + c) - c := by
      refine ciSup_le fun y ↦ ?_
      change P.expect (fun ω ↦ u (X ω + y)) - y ≤ P.oce u (fun ω ↦ X ω + c) - c
      have h : P.expect (fun ω ↦ u (X ω + c + (y - c))) - (y - c) ≤ P.oce u (fun ω ↦ X ω + c) :=
        le_ciSup (P.bddAbove_oce hux fun ω ↦ X ω + c) (y - c)
      have e : P.expect (fun ω ↦ u (X ω + c + (y - c))) = P.expect (fun ω ↦ u (X ω + y)) := by
        congr 1; funext ω; congr 1; ring
      rw [e] at h
      linarith
    linarith

/-- The OCE of a monotone utility with `u x ≤ x` is a monetary utility (paper §2.1). -/
noncomputable def toMonetaryUtility (hu : Monotone u) (hux : ∀ x, u x ≤ x) : MonetaryUtility Ω :=
  ⟨P.oce u, P.oce_mono hu hux, P.oce_cash hux⟩

end FiniteLaw

/-! ## The paper's utilities -/

/-- Entropic utility `u(x) = (1 - e^{-λx})/λ`. -/
noncomputable def entropicUtility (lam : ℝ) (x : ℝ) : ℝ := (1 - Real.exp (-lam * x)) / lam

theorem entropicUtility_le (lam : ℝ) (hl : 0 < lam) (x : ℝ) : entropicUtility lam x ≤ x := by
  unfold entropicUtility
  rw [div_le_iff₀ hl]
  have := Real.add_one_le_exp (-lam * x)
  linarith

theorem entropicUtility_monotone (lam : ℝ) (hl : 0 < lam) : Monotone (entropicUtility lam) := by
  intro a b hab
  unfold entropicUtility
  have : Real.exp (-lam * b) ≤ Real.exp (-lam * a) :=
    Real.exp_le_exp.mpr (by nlinarith)
  exact div_le_div_of_nonneg_right (by linarith) hl.le

/-- CVaR utility `u(x) = (1 + λ) min(0, x)`. -/
def cvarUtility (lam : ℝ) (x : ℝ) : ℝ := (1 + lam) * min 0 x

theorem cvarUtility_le (lam : ℝ) (hl : 0 ≤ lam) (x : ℝ) : cvarUtility lam x ≤ x := by
  unfold cvarUtility
  rcases le_or_gt 0 x with h | h
  · rw [min_eq_left h, mul_zero]; exact h
  · rw [min_eq_right h.le]; nlinarith

theorem cvarUtility_monotone (lam : ℝ) (hl : 0 ≤ lam) : Monotone (cvarUtility lam) :=
  fun _ _ hab ↦ mul_le_mul_of_nonneg_left (min_le_min_left 0 hab) (by linarith)
