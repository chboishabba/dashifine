import Mathlib

/-!
# Base-3, Base-6, and Base-9 algebraic fragments

This module packages a few small semiring/group structures that echo the
"3-6-9" story from the accompanying notes.  We work purely algebraically so the
statements can be checked inside Lean.
-/-

namespace FormalCore

/-- Base-3 numbers are modelled by natural numbers with their usual semiring
structure.  The absence of a ring structure captures the intended "no global
additive inverse" behaviour. -/
def Base3 := ℕ

instance : CommSemiring Base3 := inferInstance

theorem ternary_comm_semiring : CommSemiring Base3 := inferInstance

/-- There is no uniform additive inverse operation on `Base3`: `1` has no
partner that sums to zero. -/
lemma base3_no_global_inverse : ¬ (∀ x : Base3, ∃ y, x + y = 0) := by
  intro h
  obtain ⟨y, hy⟩ := h 1
  -- `1 + y` is a successor, so it cannot be zero.
  simpa using (Nat.succ_ne_zero y |>.trans hy)

/-- Base-6 is represented by integers modulo `6`.  This carries a ring
structure and conveniently packages four highlighted elements (representing a
simple tetralemma) inside a richer algebraic host. -/
def Base6 := ZMod 6

instance : CommRing Base6 := inferInstance
instance : Inhabited Base6 := inferInstance

/-- Four labelled states living inside `Base6`. -/
inductive TetralemmaState
  | affirm
  | deny
  | both
  | neither
  deriving DecidableEq, Repr

/-- Embed the four states as distinct residues mod `6`. -/
def encodeState : TetralemmaState → Base6
  | TetralemmaState.affirm => 0
  | TetralemmaState.deny => 2
  | TetralemmaState.both => 3
  | TetralemmaState.neither => 5

/-- A simple "return to unity" move: shift one step forward in the senary
cycle. -/
def returnToUnity (x : Base6) : Base6 := x + 1

/-- The shift stays inside the senary structure (closure). -/
lemma returnToUnity_closed (x : Base6) : ∃ y : Base6, y = returnToUnity x :=
  ⟨returnToUnity x, rfl⟩

/-- The embedding of tetralemma states is stable under the shift: the image
remains an inhabitant of `Base6`, and we can read off the shifted code. -/
lemma returnToUnity_on_states (s : TetralemmaState) : ∃ n : Base6,
    n = returnToUnity (encodeState s) :=
  ⟨returnToUnity (encodeState s), rfl⟩

/-- Base-9 is also available as an additive commutative group, useful for
tracking balanced flows. -/
def Base9 := ZMod 9

instance : AddCommGroup Base9 := inferInstance
instance : CommRing Base9 := inferInstance

end FormalCore
