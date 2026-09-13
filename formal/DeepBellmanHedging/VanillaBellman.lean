import DeepBellmanHedging.MonetaryUtility
import DeepBellmanHedging.Contraction

/-!
# The vanilla Deep Hedging Bellman equation (Theorem 3, existence and uniqueness)

Paper section 4, equation (18):
`(T̃f)(z, m) = sup_a (1/β_T(m)) U[β_T(M') f(z' + a·h', M') | m] + R̃(a; z, m)`
with the numeraire `β_T(M') = β(m) β_T(m)` and today's cashflow rewards `R̃`.
Section 5.1 shows the discounted cash-invariance `T̃(f + c) = T̃f + β(m) c`, so `T̃` is
again a `β*`-contraction.

Only existence and uniqueness are formalised. The paper's second claim, that for the
entropic or expectation utility the solution is the vanilla Deep Hedging value
function, needs the infinite-horizon time-consistency argument and is not covered.
-/

open BoundedContinuousFunction

noncomputable section

/-- Paper section 4 data: a positive numeraire that is multiplied by the discount factor
each day, normalised monetary utilities, and bounded deterministic cashflow rewards. -/
structure VanillaModel (S A Ω : Type*) where
  next : S → A → Ω → S
  cashflow : S → A → ℝ
  discount : S → ℝ
  βstar : NNReal
  discount_nonneg : ∀ s, 0 ≤ discount s
  discount_le : ∀ s, discount s ≤ βstar
  numeraire : S → ℝ
  numeraire_pos : ∀ s, 0 < numeraire s
  numeraire_next : ∀ s a ω, numeraire (next s a ω) = discount s * numeraire s
  admissible : S → Set A
  admissible_nonempty : ∀ s, (admissible s).Nonempty
  utility : S → MonetaryUtility Ω
  utility_zero : ∀ s, utility s 0 = 0
  bound : ℝ
  cashflow_bound : ∀ s, ∀ a ∈ admissible s, |cashflow s a| ≤ bound

namespace VanillaModel

variable {S A Ω : Type*} (M : VanillaModel S A Ω)

/-- `β_T(M') f(z' + a·h', M')` as a function of tomorrow's outcome. -/
def discounted (f : S → ℝ) (s : S) (a : A) : Ω → ℝ :=
  fun ω ↦ M.numeraire (M.next s a ω) * f (M.next s a ω)

/-- The action value `(1/β_T(m)) U[β_T(M') f | m] + R̃(a; z, m)`. -/
def actionValue (f : S → ℝ) (s : S) (a : A) : ℝ :=
  (M.numeraire s)⁻¹ * M.utility s (M.discounted f s a) + M.cashflow s a

/-- Paper equation (18). -/
def bellmanFun (f : S → ℝ) (s : S) : ℝ := ⨆ a : M.admissible s, M.actionValue f s a

theorem numeraire_next_nonneg (s : S) (a : A) (ω : Ω) : 0 ≤ M.numeraire (M.next s a ω) := by
  rw [M.numeraire_next]
  exact mul_nonneg (M.discount_nonneg s) (M.numeraire_pos s).le

theorem discounted_le_of_le {f g : S → ℝ} (h : ∀ x, f x ≤ g x) (s : S) (a : A) (ω : Ω) :
    M.discounted f s a ω ≤ M.discounted g s a ω :=
  mul_le_mul_of_nonneg_left (h _) (M.numeraire_next_nonneg s a ω)

section Bounded

variable [TopologicalSpace S]

/-- `|U_s[β_T(M') f]| ≤ β(s) β_T(s) ‖f‖`, using normalisation and cash-invariance. -/
theorem abs_utility_discounted_le (f : S →ᵇ ℝ) (s : S) (a : A) :
    |M.utility s (M.discounted f s a)| ≤ M.discount s * M.numeraire s * ‖f‖ := by
  have hf : ∀ x, |f x| ≤ ‖f‖ := fun x ↦ by simpa [Real.norm_eq_abs] using norm_coe_le_norm f x
  have hn : ∀ ω, M.numeraire (M.next s a ω) = M.discount s * M.numeraire s :=
    fun ω ↦ M.numeraire_next s a ω
  have hpos : 0 ≤ M.discount s * M.numeraire s :=
    mul_nonneg (M.discount_nonneg s) (M.numeraire_pos s).le
  have upper : M.utility s (M.discounted f s a)
      ≤ M.utility s 0 + M.discount s * M.numeraire s * ‖f‖ := by
    refine (M.utility s).le_add_of_le fun ω ↦ ?_
    simp only [discounted, hn, Pi.zero_apply, zero_add]
    exact mul_le_mul_of_nonneg_left ((le_abs_self _).trans (hf _)) hpos
  have lower : M.utility s 0 - M.discount s * M.numeraire s * ‖f‖
      ≤ M.utility s (M.discounted f s a) := by
    refine (M.utility s).add_le_of_le fun ω ↦ ?_
    simp only [discounted, hn, Pi.zero_apply, zero_sub]
    have h1 : -‖f‖ ≤ f (M.next s a ω) := (neg_le_neg (hf _)).trans (neg_abs_le _)
    have := mul_le_mul_of_nonneg_left h1 hpos
    rwa [mul_neg] at this
  rw [M.utility_zero] at upper lower
  rw [abs_le]
  constructor <;> linarith

theorem abs_actionValue_le (f : S →ᵇ ℝ) (s : S) (a : A) (ha : a ∈ M.admissible s) :
    |M.actionValue f s a| ≤ M.bound + M.βstar * ‖f‖ := by
  have hpos := M.numeraire_pos s
  have hd := abs_le.mp (M.abs_utility_discounted_le f s a)
  have hc := abs_le.mp (M.cashflow_bound s a ha)
  have hβ : M.discount s * ‖f‖ ≤ M.βstar * ‖f‖ :=
    mul_le_mul_of_nonneg_right (M.discount_le s) (norm_nonneg f)
  have hne := hpos.ne'
  have hinv : (M.numeraire s)⁻¹ * (M.discount s * M.numeraire s * ‖f‖) = M.discount s * ‖f‖ := by
    field_simp
  have key : |(M.numeraire s)⁻¹ * M.utility s (M.discounted f s a)| ≤ M.discount s * ‖f‖ := by
    rw [abs_mul, abs_of_pos (inv_pos.mpr hpos), ← hinv]
    exact mul_le_mul_of_nonneg_left (abs_le.mpr hd) (inv_pos.mpr hpos).le
  have := abs_le.mp key
  unfold actionValue
  rw [abs_le]
  constructor <;> linarith

theorem bddAbove_actionValue (f : S →ᵇ ℝ) (s : S) :
    BddAbove (Set.range fun a : M.admissible s ↦ M.actionValue f s a) := by
  refine ⟨M.bound + M.βstar * ‖f‖, ?_⟩
  rintro _ ⟨a, rfl⟩
  exact (le_abs_self _).trans (M.abs_actionValue_le f s a a.2)

theorem abs_bellmanFun_le (f : S →ᵇ ℝ) (s : S) :
    |M.bellmanFun f s| ≤ M.bound + M.βstar * ‖f‖ := by
  have := (M.admissible_nonempty s).to_subtype
  rw [abs_le]
  constructor
  · obtain ⟨a⟩ := (M.admissible_nonempty s).to_subtype
    have h1 := (abs_le.mp (M.abs_actionValue_le f s a a.2)).1
    exact h1.trans (le_ciSup (M.bddAbove_actionValue f s) a)
  · exact ciSup_le fun a ↦ (le_abs_self _).trans (M.abs_actionValue_le f s a a.2)

end Bounded

section Operator

variable [TopologicalSpace S] [DiscreteTopology S]

/-- The vanilla operator on bounded functions. -/
def bellman (f : S →ᵇ ℝ) : S →ᵇ ℝ :=
  mkOfBound ⟨M.bellmanFun f, continuous_of_discreteTopology⟩ (2 * (M.bound + M.βstar * ‖f‖))
    fun x y ↦ by
      rw [Real.dist_eq]
      change |M.bellmanFun f x - M.bellmanFun f y| ≤ _
      calc |M.bellmanFun f x - M.bellmanFun f y|
          ≤ |M.bellmanFun f x| + |M.bellmanFun f y| := abs_sub _ _
        _ ≤ _ := by linarith [M.abs_bellmanFun_le f x, M.abs_bellmanFun_le f y]

@[simp] theorem bellman_apply (f : S →ᵇ ℝ) (s : S) : M.bellman f s = M.bellmanFun f s := rfl

theorem bellman_monotone : Monotone M.bellman := by
  intro f g hfg s
  change M.bellmanFun f s ≤ M.bellmanFun g s
  have := (M.admissible_nonempty s).to_subtype
  refine ciSup_mono (M.bddAbove_actionValue g s) fun a ↦ ?_
  unfold actionValue
  have hU : M.utility s (M.discounted f s a) ≤ M.utility s (M.discounted g s a) :=
    (M.utility s).mono fun ω ↦ M.discounted_le_of_le (fun x ↦ hfg x) s a ω
  have hinv := (inv_pos.mpr (M.numeraire_pos s)).le
  linarith [mul_le_mul_of_nonneg_left hU hinv]

/-- Discounted cash-invariance (paper section 5.1): `T̃(f + c) = T̃f + β(m) c ≤ T̃f + β* c`. -/
theorem bellman_cash (f : S →ᵇ ℝ) (c : ℝ) (hc : 0 ≤ c) :
    M.bellman (f + const S c) ≤ M.bellman f + const S (M.βstar * c) := by
  intro s
  change M.bellmanFun (f + const S c) s ≤ M.bellmanFun f s + M.βstar * c
  have := (M.admissible_nonempty s).to_subtype
  refine ciSup_le fun a ↦ ?_
  have hpos := M.numeraire_pos s
  have hne := hpos.ne'
  have hshift : M.discounted (f + const S c) s a
      = fun ω ↦ M.discounted f s a ω + M.discount s * M.numeraire s * c := by
    funext ω
    simp only [discounted, const_apply, Pi.add_apply, M.numeraire_next]
    ring
  have hβ := mul_le_mul_of_nonneg_right (M.discount_le s) hc
  have hval : M.actionValue (f + const S c) s a = M.actionValue f s a + M.discount s * c := by
    unfold actionValue
    rw [hshift, (M.utility s).cash]
    field_simp
    ring
  calc M.actionValue (f + const S c) s a = M.actionValue f s a + M.discount s * c := hval
    _ ≤ M.bellmanFun f s + M.βstar * c := by
        unfold bellmanFun
        exact add_le_add (le_ciSup (M.bddAbove_actionValue f s) a) hβ

/-- **Theorem 3** (`th:convergence_dh`, existence and uniqueness part). -/
theorem existsUnique_value (hβ : M.βstar < 1) : ∃! V : S →ᵇ ℝ, M.bellman V = V :=
  existsUnique_fixedPoint_of_monotone_cash hβ M.bellman_monotone M.bellman_cash

end Operator

end VanillaModel

end
