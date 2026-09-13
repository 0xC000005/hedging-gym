import Mathlib.Order.Monotone.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Monetary utilities

Buehler, Murray and Wood, *Deep Bellman Hedging* (arXiv:2207.00932), section 2.1:
a monetary utility is monotone and cash-invariant. Normalisation and concavity are
part of the paper's definition but are not used by the fixed-point theorems, so
they are omitted here; `OCE.lean` shows that optimized certainty equivalents are
monetary utilities in this sense.
-/

/-- A monetary utility on outcomes `Ω → ℝ`: monotone and cash-invariant (paper §2.1). -/
structure MonetaryUtility (Ω : Type*) where
  toFun : (Ω → ℝ) → ℝ
  mono : Monotone toFun
  cash : ∀ (X : Ω → ℝ) (c : ℝ), toFun (fun ω ↦ X ω + c) = toFun X + c

namespace MonetaryUtility

variable {Ω : Type*} (U : MonetaryUtility Ω)

instance : CoeFun (MonetaryUtility Ω) (fun _ ↦ (Ω → ℝ) → ℝ) := ⟨MonetaryUtility.toFun⟩

/-- Monotonicity and cash-invariance combined: `X ≤ Y + c` pointwise gives `U X ≤ U Y + c`. -/
theorem le_add_of_le {X Y : Ω → ℝ} {c : ℝ} (h : ∀ ω, X ω ≤ Y ω + c) : U X ≤ U Y + c := by
  calc U X ≤ U (fun ω ↦ Y ω + c) := U.mono (fun ω ↦ h ω)
    _ = U Y + c := U.cash Y c

/-- The lower-bound counterpart of `le_add_of_le`. -/
theorem add_le_of_le {X Y : Ω → ℝ} {c : ℝ} (h : ∀ ω, Y ω - c ≤ X ω) : U Y - c ≤ U X := by
  have := U.le_add_of_le (X := Y) (Y := X) (c := c) (fun ω ↦ by linarith [h ω])
  linarith

end MonetaryUtility
