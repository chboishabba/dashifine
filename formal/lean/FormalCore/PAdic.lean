import Mathlib

/-!
# 3-adic geometric series

This file records a concrete calculation in the 3-adic numbers: the infinite
series `1 + 3 + 3^2 + …` converges in `ℚ_[3]` and sums to `-1/2`.

The statement mirrors the informal computation in the accompanying PDF: the
p-adic norm of `3` is strictly less than `1`, so the usual geometric-series
formula applies.
-/-

open scoped BigOperators

namespace FormalCore

/-- The 3-adic absolute value of `3` is `1/3`, so it sits inside the geometric
series radius of convergence. -/
lemma norm_three_padic_lt_one : ‖(3 : ℚ_[3])‖ < 1 := by
  -- In `ℚ_[p]` the norm of `p` is `p⁻¹`.
  have hnorm : ‖(3 : ℚ_[3])‖ = (3 : ℝ)⁻¹ := by
    simpa using (padicNormE.norm_p (p := 3))
  -- And `1 / 3 < 1` in the reals.
  have : (3 : ℝ)⁻¹ < 1 := by norm_num
  simpa [hnorm] using this

/-- The geometric series with ratio `3` converges to `-1/2` in the 3-adics. -/
theorem geom_sum_3adic : (∑' n : ℕ, (3 : ℚ_[3]) ^ n) = (- (1 / 2 : ℚ_[3])) := by
  -- Summability follows from the norm bound.
  have hsum := tsum_geometric_of_norm_lt_1 (r := (3 : ℚ_[3])) norm_three_padic_lt_one
  -- The geometric closed form is `1 / (1 - r)`.
  have hgeom : (1 : ℚ_[3]) - 3 ≠ 0 := by norm_num
  -- Rewrite the expression to match the target `-1/2`.
  have hval : (1 : ℚ_[3]) / ((1 : ℚ_[3]) - 3) = - (1 / 2 : ℚ_[3]) := by
    field_simp [hgeom, sub_eq_add_neg]
  simpa [hgeom, hval] using hsum

/-- The ratio `9 = 3^2` still has norm strictly below one. -/
lemma norm_nine_padic_lt_one : ‖(9 : ℚ_[3])‖ < 1 := by
  have hnorm : ‖(9 : ℚ_[3])‖ = ‖(3 : ℚ_[3])‖ ^ 2 := by
    have hpow : (9 : ℚ_[3]) = (3 : ℚ_[3]) ^ 2 := by norm_num
    simpa [hpow] using (norm_pow (3 : ℚ_[3]) 2)
  have hpow : ‖(3 : ℚ_[3])‖ ^ 2 < 1 := by
    have hnonneg : 0 ≤ ‖(3 : ℚ_[3])‖ := norm_nonneg _
    have hlt : ‖(3 : ℚ_[3])‖ < 1 := norm_three_padic_lt_one
    have : 0 < (2 : ℕ) := by decide
    exact pow_lt_one hnonneg hlt this
  simpa [hnorm] using hpow

/-- A slightly larger ratio also fits in the p-adic geometric sum toolkit. -/
theorem geom_sum_3adic_squared : (∑' n : ℕ, (9 : ℚ_[3]) ^ n) = (- (1 / 8 : ℚ_[3])) := by
  have hsum := tsum_geometric_of_norm_lt_1 (r := (9 : ℚ_[3])) norm_nine_padic_lt_one
  have hgeom : (1 : ℚ_[3]) - 9 ≠ 0 := by norm_num
  have hval : (1 : ℚ_[3]) / ((1 : ℚ_[3]) - 9) = - (1 / 8 : ℚ_[3]) := by
    field_simp [hgeom, sub_eq_add_neg]
  simpa [hgeom, hval] using hsum

end FormalCore
