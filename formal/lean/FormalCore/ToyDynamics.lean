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

/-- Shorthand for describing a path that keeps finding a `safe` suffix. -/
def SafePath (s : DynState) : Prop := ∀ n, ∃ m, iterate (n + m) s = DynState.safe

@[simp] lemma step_safe : step DynState.safe = DynState.safe := rfl

@[simp] lemma iterate_safe (n : ℕ) : iterate n DynState.safe = DynState.safe := by
  induction n with
  | zero => rfl
  | succ n ih => simpa [iterate_succ, step_safe] using ih

lemma iterate_add (a b : ℕ) (s : DynState) : iterate (a + b) s = iterate b (iterate a s) := by
  simpa [iterate] using (Nat.iterate_add (f := step) a b s)

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

/-- A concrete temporal-style property: every suffix of a run eventually lands in `safe`. -/
lemma safePath_of_reaches (s : DynState) {k : ℕ} (h : iterate k s = DynState.safe) : SafePath s := by
  intro n
  refine ⟨k, ?_⟩
  have hk : iterate (k + n) s = iterate n (iterate k s) := by
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using iterate_add k n s
  simpa [hk, h] using iterate_safe n

/-- The property `SafePath` is preserved by a single system step. -/
lemma safePath_step (s : DynState) (h : SafePath s) : SafePath (step s) := by
  intro n
  obtain ⟨m, hm⟩ := h (n + 1)
  refine ⟨m, ?_⟩
  have hstep : iterate (n + m) (step s) = iterate (n + m + 1) s := by
    have := iterate_succ (n := n + m) (s := s)
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using this.symm
  have hm' : iterate (n + m + 1) s = DynState.safe := by
    simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hm
  exact hstep.trans hm'

/-- The entire system enjoys the temporal "always eventually safe" behaviour. -/
theorem system_safe_paths : ∀ s, SafePath s := by
  intro s
  obtain ⟨k, hk⟩ := system_always_eventually_safe s
  exact safePath_of_reaches s hk

end FormalCore
