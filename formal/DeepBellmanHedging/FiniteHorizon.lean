import DeepBellmanHedging.Bellman

/-!
# Finite horizon: the setting hedging-gym implements

The Python baseline uses zero rates (`β = 1`) and a liability that settles at a fixed
horizon `N`, with the continuation value masked to zero at the final decision. The
contraction argument of Theorem 1 does not apply with `β = 1`; instead the operator
only looks one step ahead in time and ignores its input at or after the horizon,
so `T^[N+1]` does not depend on its argument at all. This gives a unique fixed point
by backward induction, reached by value iteration in `N + 1` sweeps.
-/

open BoundedContinuousFunction

variable {S : Type*} [TopologicalSpace S]

/-- An operator that only depends on its input at strictly later times and ignores it
from the horizon on has a unique fixed point, reached in `N + 1` iterations. -/
theorem existsUnique_fixedPoint_of_horizon {T : (S →ᵇ ℝ) → (S →ᵇ ℝ)} (time : S → ℕ) (N : ℕ)
    (local_dep : ∀ f g s, (∀ x, time s < time x → f x = g x) → T f s = T g s)
    (terminal : ∀ f g s, N ≤ time s → T f s = T g s) :
    (∃! V : S →ᵇ ℝ, T V = V) ∧ ∀ f g : S →ᵇ ℝ, T^[N + 1] f = T^[N + 1] g := by
  have iter : ∀ k (f g : S →ᵇ ℝ) s, N ≤ time s + k → T^[k + 1] f s = T^[k + 1] g s := by
    intro k
    induction k with
    | zero => intro f g s h; simpa using terminal f g s (by simpa using h)
    | succ k ih =>
      intro f g s h
      rw [Function.iterate_succ_apply' T (k + 1) f, Function.iterate_succ_apply' T (k + 1) g]
      exact local_dep _ _ s fun x hx ↦ ih f g x (by omega)
  have hN : ∀ f g : S →ᵇ ℝ, T^[N + 1] f = T^[N + 1] g := fun f g ↦
    ext fun s ↦ iter N f g s (by omega)
  have contracting : ContractingWith 0 T^[N + 1] :=
    ⟨by norm_num, LipschitzWith.of_dist_le_mul fun f g ↦ by rw [hN f g, dist_self]; simp⟩
  refine ⟨⟨_, contracting.isFixedPt_fixedPoint_iterate, fun W hW ↦ ?_⟩, hN⟩
  have hV := contracting.isFixedPt_fixedPoint_iterate
  calc W = T^[N + 1] W := (Function.IsFixedPt.iterate hW (N + 1)).symm
    _ = T^[N + 1] (ContractingWith.fixedPoint T^[N + 1] contracting) := hN _ _
    _ = _ := hV.iterate (N + 1)

namespace HedgingModel

variable {S A Ω : Type*} [TopologicalSpace S] [DiscreteTopology S] (M : HedgingModel S A Ω)

/-- The Bellman operator only reads `f` one step ahead in time. -/
theorem bellman_local (time : S → ℕ) (htime : ∀ s a ω, time (M.next s a ω) = time s + 1)
    (f g : S →ᵇ ℝ) (s : S) (h : ∀ x, time s < time x → f x = g x) :
    M.bellman f s = M.bellman g s := by
  simp only [bellman_apply, bellmanFun]
  refine iSup_congr fun a ↦ congrArg (M.utility s) (funext fun ω ↦ ?_)
  simp only [step]
  rw [h (M.next s a ω) (by rw [htime]; omega)]

/-- With the continuation masked at the horizon, the operator ignores `f` there. -/
theorem bellman_terminal (time : S → ℕ) (N : ℕ) (hmask : ∀ s, N ≤ time s → M.discount s = 0)
    (f g : S →ᵇ ℝ) (s : S) (hs : N ≤ time s) : M.bellman f s = M.bellman g s := by
  simp only [bellman_apply, bellmanFun]
  refine iSup_congr fun a ↦ congrArg (M.utility s) (funext fun ω ↦ ?_)
  simp [step, hmask s hs]

/-- **Finite-horizon Deep Bellman Hedging**: zero rates (`β ≤ 1`) with the continuation
value masked from the horizon on give a unique bounded solution, and value iteration
reaches it in `N + 1` sweeps from any start. -/
theorem existsUnique_value_of_horizon (time : S → ℕ) (N : ℕ)
    (htime : ∀ s a ω, time (M.next s a ω) = time s + 1)
    (hmask : ∀ s, N ≤ time s → M.discount s = 0) :
    (∃! V : S →ᵇ ℝ, M.bellman V = V) ∧ ∀ f g : S →ᵇ ℝ, M.bellman^[N + 1] f = M.bellman^[N + 1] g :=
  existsUnique_fixedPoint_of_horizon time N (M.bellman_local time htime)
    (M.bellman_terminal time N hmask)

end HedgingModel
