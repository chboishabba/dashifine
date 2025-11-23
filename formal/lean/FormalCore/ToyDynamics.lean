import Mathlib

/-!
# A toy transition system

We build a tiny state graph and prove that every execution eventually lands in a
`safe` node.  This is a minimal analogue of a μ-calculus / temporal-logic
"always eventually safe" specification.
-/-

namespace FormalCore

/-- Five labelled states for the transition system. -/
inductive DynState
  | start
  | branch
  | overflow
  | retry
  | safe
  deriving DecidableEq, Repr

/-- One-step transition function.  Every path is steered toward `safe`. -/
def step : DynState → DynState
  | DynState.start => DynState.branch
  | DynState.branch => DynState.safe
  | DynState.overflow => DynState.safe
  | DynState.retry => DynState.overflow
  | DynState.safe => DynState.safe

/-- Iterate the transition function `n` times. -/
def iterate (n : ℕ) (s : DynState) : DynState := Nat.iterate step n s

@[simp] lemma iterate_zero (s : DynState) : iterate 0 s = s := rfl
@[simp] lemma iterate_succ (n : ℕ) (s : DynState) : iterate (n.succ) s = iterate n (step s) := by
  simp [iterate, Nat.iterate]

/-- The system inevitably arrives at the `safe` state within two steps. -/
lemma reaches_safe (s : DynState) : ∃ n ≤ 2, iterate n s = DynState.safe := by
  classical
  cases s <;> refine ?_ <;> decide
  · -- start
    refine ⟨2, by decide, ?_⟩
    decide
  · -- branch
    refine ⟨1, by decide, ?_⟩
    decide
  · -- overflow
    refine ⟨1, by decide, ?_⟩
    decide
  · -- retry
    refine ⟨2, by decide, ?_⟩
    decide
  · -- safe
    refine ⟨0, by decide, rfl⟩

/-- A simple "always eventually safe" property: from every state, some suffix
of the execution hits `safe`. -/
def AlwaysEventuallySafe : Prop := ∀ s, ∃ n, iterate n s = DynState.safe

/-- The toy system satisfies the safety eventuality. -/
theorem system_always_eventually_safe : AlwaysEventuallySafe := by
  intro s
  obtain ⟨n, _, h⟩ := reaches_safe s
  exact ⟨n, h⟩

end FormalCore
