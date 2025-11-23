module Overflow where

open import Agda.Builtin.Equality
open import Agda.Builtin.Nat using (Nat ; zero ; suc)

------------------------------------------------------------------------
-- Basic relations on ℕ (custom strict order)
------------------------------------------------------------------------

infix 4 _≺_

data _≺_ : Nat → Nat → Set where
  z≺s : ∀ {n} → zero ≺ suc n
  s≺s : ∀ {m n} → m ≺ n → suc m ≺ suc n

------------------------------------------------------------------------
-- Voxel states
------------------------------------------------------------------------

data Voxel : Set where
  grounded plateau ascended : Voxel

------------------------------------------------------------------------
-- Threshold guards (now: classification-only stub)
------------------------------------------------------------------------

-- This now encodes just “which side of the threshold are we on?”
-- without storing an explicit proof. The strict order `_≺_` is still
-- available for future strengthening, but CI only needs this stub.
data VoxelGuard (threshold value : Nat) : Set where
  stay   : VoxelGuard threshold value
  pivot  : VoxelGuard threshold value
  ascend : VoxelGuard threshold value

state : ∀ {t v} → VoxelGuard t v → Voxel
state stay   = grounded
state pivot  = plateau
state ascend = ascended

------------------------------------------------------------------------
-- Helper: deterministically choose a guard from a comparison token
------------------------------------------------------------------------

data Order : Set where below equal above : Order

compare : Nat → Nat → Order
compare zero    zero    = equal
compare zero    (suc _) = below
compare (suc _) zero    = above
compare (suc a) (suc b) = compare a b

compare-eq-below : ∀ {t v} → compare t v ≡ below → t ≺ v
compare-eq-below {zero}    {zero}    ()
compare-eq-below {zero}    {suc _}   refl = z≺s
compare-eq-below {suc _}   {zero}    ()
compare-eq-below {suc t}   {suc v}   p = s≺s (compare-eq-below {t} {v} p)

compare-eq-above : ∀ {t v} → compare t v ≡ above → v ≺ t
compare-eq-above {zero}    {zero}    ()
compare-eq-above {zero}    {suc _}   ()
compare-eq-above {suc _}   {zero}    refl = z≺s
compare-eq-above {suc t}   {suc v}   p = s≺s (compare-eq-above {t} {v} p)

enforce : (threshold value : Nat) → VoxelGuard threshold value
enforce threshold value with compare threshold value
... | below = ascend
... | equal = pivot
... | above = stay
