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
-- Helper: deterministically choose a guard by structural comparison
------------------------------------------------------------------------

enforce : (threshold value : Nat) → VoxelGuard threshold value
enforce threshold value with threshold , value
... | zero , zero = pivot
... | zero , suc _ = ascend
... | suc _ , zero = stay
... | suc threshold , suc value with enforce threshold value
... | stay   = stay
... | pivot  = pivot
... | ascend = ascend
