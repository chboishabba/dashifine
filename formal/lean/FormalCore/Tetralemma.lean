import Mathlib

/-!
# A tiny probabilistic tetralemma

We model a 2×2 binary grid as `Bool × Bool`.  A uniform prior assigns
probability `1/4` to each outcome, and marginalising across one axis yields a
`1/2` probability—matching the "0.25 attention / 0.5 branch" thresholds.
-/-

open scoped BigOperators

namespace FormalCore

/-- Outcomes in the 2×2 grid. -/
abbrev Outcome := Bool × Bool

/-- Uniform mass over the four cells. -/
def uniformOutcome (_ : Outcome) : ℝ := (1 / 4 : ℝ)

lemma uniform_value (o : Outcome) : uniformOutcome o = (1 / 4 : ℝ) := rfl

/-- Summing the uniform prior over the whole grid yields total mass `1`. -/
lemma uniform_total_mass : (∑ o : Outcome, uniformOutcome o) = 1 := by
  classical
  -- Sum of a constant over a finite type is its cardinality times the constant.
  have hsum : (∑ _o : Outcome, (1 / 4 : ℝ)) = (Fintype.card Outcome) • (1 / 4 : ℝ) := by
    simpa using (Finset.sum_const (a := (1 / 4 : ℝ)) (s := (Finset.univ : Finset Outcome)))
  -- There are four outcomes in the grid.
  have hcard : Fintype.card Outcome = 4 := by decide
  -- Rewrite the scalar multiplication as usual multiplication.
  have hmul : ((Fintype.card Outcome) • (1 / 4 : ℝ)) = (4 : ℝ) * (1 / 4 : ℝ) := by
    simpa [hcard, nsmul_eq_mul]
  -- Put the pieces together.
  have : (∑ _o : Outcome, (1 / 4 : ℝ)) = 1 := by
    simpa [hmul] using hsum
  simpa [uniformOutcome] using this

/-- Marginalising over one axis leaves probability `1/2`, the branching
threshold. -/
lemma axis_marginal (a : Bool) : (∑ b : Bool, uniformOutcome (a, b)) = (1 / 2 : ℝ) := by
  classical
  have hsum : (∑ _b : Bool, (1 / 4 : ℝ)) = (Finset.card (Finset.univ : Finset Bool)) • (1 / 4 : ℝ) := by
    simpa using (Finset.sum_const (a := (1 / 4 : ℝ)) (s := (Finset.univ : Finset Bool)))
  have hcard : (Finset.card (Finset.univ : Finset Bool)) = 2 := by decide
  have hmul : ((Finset.card (Finset.univ : Finset Bool)) • (1 / 4 : ℝ)) = (1 / 2 : ℝ) := by
    simpa [hcard, nsmul_eq_mul] using (by decide : ((2) • (1 / 4 : ℝ)) = (1 / 2 : ℝ))
  simpa [uniformOutcome] using hsum.trans hmul

/-- Because the marginal already meets the `0.5` threshold, we can interpret it
as a definite branch selection on the chosen axis. -/
lemma branch_threshold (a : Bool) : (0.5 : ℝ) ≤ ∑ b : Bool, uniformOutcome (a, b) := by
  have h := axis_marginal a
  have : (0.5 : ℝ) ≤ (1 / 2 : ℝ) := by norm_num
  simpa [h] using this

end FormalCore
