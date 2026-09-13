import DeepBellmanHedging.MonetaryUtility
import DeepBellmanHedging.Contraction

/-!
# The Deep Bellman Hedging equation (Theorem 1)

Paper section 2: states `s = (z, m)` with portfolio `z` and market `m`, actions `a` in
a state-dependent admissible set, tomorrow's market `M'` as an outcome `ω`. The
Bellman operator is `(Tf)(s) = sup_a U_s[ β(s) f(next(s, a, ω)) + R(a; s, ω) ]`
(paper equation (3)). With bounded `f`, monotonicity and cash-invariance of `U_s`
make `T` a `β*`-contraction, so the equation has a unique bounded solution.

The paper's "rewards are finite" hypothesis (equation (4)) is pointwise in `s`; the
sup-norm argument needs `T0` itself to be bounded, so `reward_bound` is stated
uniformly in `s`. See `formal/README.md`.
-/

open BoundedContinuousFunction

noncomputable section

/-- Paper section 2 data. `βstar` bounds the discount factor; Theorem 1 needs `βstar < 1`,
the finite-horizon variant allows `βstar = 1`. -/
structure HedgingModel (S A Ω : Type*) where
  next : S → A → Ω → S
  reward : S → A → Ω → ℝ
  discount : S → ℝ
  βstar : NNReal
  discount_nonneg : ∀ s, 0 ≤ discount s
  discount_le : ∀ s, discount s ≤ βstar
  admissible : S → Set A
  admissible_nonempty : ∀ s, (admissible s).Nonempty
  utility : S → MonetaryUtility Ω
  bound : ℝ
  reward_bound : ∀ s, ∀ a ∈ admissible s, |utility s (reward s a)| ≤ bound

namespace HedgingModel

variable {S A Ω : Type*} (M : HedgingModel S A Ω)

/-- `β(m) f(z' + a·h', M') + R(a; z, m, M')` as a function of tomorrow's outcome. -/
def step (f : S → ℝ) (s : S) (a : A) : Ω → ℝ :=
  fun ω ↦ M.discount s * f (M.next s a ω) + M.reward s a ω

/-- `(Tf)(z, m) = sup_a U[β f(z' + a·h', M') + R | m]` over admissible actions (paper (3)). -/
def bellmanFun (f : S → ℝ) (s : S) : ℝ :=
  ⨆ a : M.admissible s, M.utility s (M.step f s a)

theorem step_le_of_le {f g : S → ℝ} (h : ∀ x, f x ≤ g x) (s : S) (a : A) (ω : Ω) :
    M.step f s a ω ≤ M.step g s a ω := by
  unfold step
  have := mul_le_mul_of_nonneg_left (h (M.next s a ω)) (M.discount_nonneg s)
  linarith

section Bounded

variable [TopologicalSpace S]

/-- Bounded inputs give uniformly bounded integrands: `|U_s[step f]| ≤ bound + β* ‖f‖`. -/
theorem abs_utility_step_le (f : S →ᵇ ℝ) (s : S) (a : A) (ha : a ∈ M.admissible s) :
    |M.utility s (M.step f s a)| ≤ M.bound + M.βstar * ‖f‖ := by
  have hf : ∀ x, |f x| ≤ ‖f‖ := fun x ↦ by simpa [Real.norm_eq_abs] using norm_coe_le_norm f x
  have hβ : M.discount s * ‖f‖ ≤ M.βstar * ‖f‖ :=
    mul_le_mul_of_nonneg_right (M.discount_le s) (norm_nonneg f)
  have upper : M.utility s (M.step f s a) ≤ M.utility s (M.reward s a) + M.discount s * ‖f‖ := by
    refine (M.utility s).le_add_of_le fun ω ↦ ?_
    unfold step
    have := mul_le_mul_of_nonneg_left ((le_abs_self _).trans (hf (M.next s a ω)))
      (M.discount_nonneg s)
    linarith
  have lower : M.utility s (M.reward s a) - M.discount s * ‖f‖ ≤ M.utility s (M.step f s a) := by
    refine (M.utility s).add_le_of_le fun ω ↦ ?_
    unfold step
    have h1 : -‖f‖ ≤ f (M.next s a ω) := (neg_le_neg (hf _)).trans (neg_abs_le _)
    have := mul_le_mul_of_nonneg_left h1 (M.discount_nonneg s)
    rw [mul_neg] at this
    linarith
  have hb := abs_le.mp (M.reward_bound s a ha)
  rw [abs_le]
  constructor <;> linarith [hb.1, hb.2]

theorem bddAbove_step (f : S →ᵇ ℝ) (s : S) :
    BddAbove (Set.range fun a : M.admissible s ↦ M.utility s (M.step f s a)) := by
  refine ⟨M.bound + M.βstar * ‖f‖, ?_⟩
  rintro _ ⟨a, rfl⟩
  exact (le_abs_self _).trans (M.abs_utility_step_le f s a a.2)

theorem abs_bellmanFun_le (f : S →ᵇ ℝ) (s : S) :
    |M.bellmanFun f s| ≤ M.bound + M.βstar * ‖f‖ := by
  have := (M.admissible_nonempty s).to_subtype
  rw [abs_le]
  constructor
  · obtain ⟨a⟩ := (M.admissible_nonempty s).to_subtype
    have h1 := (abs_le.mp (M.abs_utility_step_le f s a a.2)).1
    exact h1.trans (le_ciSup (M.bddAbove_step f s) a)
  · exact ciSup_le fun a ↦ (le_abs_self _).trans (M.abs_utility_step_le f s a a.2)

end Bounded

section Operator

variable [TopologicalSpace S] [DiscreteTopology S]

/-- The Bellman operator on bounded functions; `S` carries the discrete topology. -/
def bellman (f : S →ᵇ ℝ) : S →ᵇ ℝ :=
  mkOfBound ⟨M.bellmanFun f, continuous_of_discreteTopology⟩ (2 * (M.bound + M.βstar * ‖f‖))
    fun x y ↦ by
      rw [Real.dist_eq]
      change |M.bellmanFun f x - M.bellmanFun f y| ≤ _
      calc |M.bellmanFun f x - M.bellmanFun f y|
          ≤ |M.bellmanFun f x| + |M.bellmanFun f y| := abs_sub _ _
        _ ≤ _ := by linarith [M.abs_bellmanFun_le f x, M.abs_bellmanFun_le f y]

@[simp] theorem bellman_apply (f : S →ᵇ ℝ) (s : S) : M.bellman f s = M.bellmanFun f s := rfl

/-- Monotonicity of `U_s` and `β ≥ 0` make `T` monotone (paper section 5). -/
theorem bellman_monotone : Monotone M.bellman := by
  intro f g hfg s
  change M.bellmanFun f s ≤ M.bellmanFun g s
  have := (M.admissible_nonempty s).to_subtype
  exact ciSup_mono (M.bddAbove_step g s) fun a ↦
    (M.utility s).mono fun ω ↦ M.step_le_of_le (fun x ↦ hfg x) s a ω

/-- Cash-invariance of `U_s` gives `T(f + c) ≤ Tf + β* c` (paper section 5). -/
theorem bellman_cash (f : S →ᵇ ℝ) (c : ℝ) (hc : 0 ≤ c) :
    M.bellman (f + const S c) ≤ M.bellman f + const S (M.βstar * c) := by
  intro s
  change M.bellmanFun (f + const S c) s ≤ M.bellmanFun f s + M.βstar * c
  have := (M.admissible_nonempty s).to_subtype
  refine ciSup_le fun a ↦ ?_
  have hstep : M.step (f + const S c) s a = fun ω ↦ M.step f s a ω + M.discount s * c := by
    funext ω
    simp only [step, const_apply, Pi.add_apply]
    ring
  have hβ := mul_le_mul_of_nonneg_right (M.discount_le s) hc
  calc M.utility s (M.step (f + const S c) s a)
      = M.utility s (M.step f s a) + M.discount s * c := by rw [hstep, (M.utility s).cash]
    _ ≤ M.bellmanFun f s + M.βstar * c := by
        unfold bellmanFun
        exact add_le_add (le_ciSup (M.bddAbove_step f s) a) hβ

/-- **Theorem 1** (`th:convergence`): with `β* < 1` the Bellman equation has a unique
bounded solution. -/
theorem existsUnique_value (hβ : M.βstar < 1) : ∃! V : S →ᵇ ℝ, M.bellman V = V :=
  existsUnique_fixedPoint_of_monotone_cash hβ M.bellman_monotone M.bellman_cash

/-- Value iteration converges to the solution from any bounded start (footnote 18). -/
theorem tendsto_iterate_value (hβ : M.βstar < 1) (V₀ : S →ᵇ ℝ) :
    ∃ V : S →ᵇ ℝ, M.bellman V = V ∧
      Filter.Tendsto (fun n ↦ M.bellman^[n] V₀) Filter.atTop (nhds V) :=
  tendsto_iterate_of_monotone_cash hβ M.bellman_monotone M.bellman_cash V₀

end Operator

end HedgingModel

end
