module Overflow where

open import Agda.Builtin.Equality
open import Agda.Builtin.Nat

------------------------------------------------------------------------
-- Basic relations on ℕ
------------------------------------------------------------------------

data _<_ : Nat → Nat → Set where
  z<s : ∀ {n} → zero < suc n
  s<s : ∀ {m n} → m < n → suc m < suc n

------------------------------------------------------------------------
-- Voxel states
------------------------------------------------------------------------

data Voxel : Set where
  grounded plateau ascended : Voxel

------------------------------------------------------------------------
-- Threshold guards
------------------------------------------------------------------------

-- The guard forces an "ascended" voxel whenever a proof of overflow is provided.
data VoxelGuard : (threshold value : Nat) → Set where
  stay   : value < threshold      → VoxelGuard threshold value
  pivot  : threshold ≡ value      → VoxelGuard threshold value
  ascend : threshold < value      → VoxelGuard threshold value

state : ∀ {t v} → VoxelGuard t v → Voxel
state (stay _)   = grounded
state (pivot _)  = plateau
state (ascend _) = ascended

------------------------------------------------------------------------
-- Helper: deterministically choose a guard from a comparison token
------------------------------------------------------------------------

data Order : Set where below equal above : Order

compare : Nat → Nat → Order
compare zero    zero    = equal
compare zero    (suc _) = below
compare (suc _) zero    = above
compare (suc a) (suc b) = compare a b

ltFromCompare : ∀ {t v} → compare t v ≡ above → v < t
ltFromCompare {zero}    {zero}    ()
ltFromCompare {zero}    {suc _}   ()
ltFromCompare {suc _}   {zero}    _  = z<s
ltFromCompare {suc t}   {suc v}   pf = s<s (ltFromCompare {t} {v} pf)

ltFromBelow : ∀ {t v} → compare t v ≡ below → t < v
ltFromBelow {zero}    {zero}    ()
ltFromBelow {zero}    {suc _}   _  = z<s
ltFromBelow {suc _}   {zero}    ()
ltFromBelow {suc t}   {suc v}   pf = s<s (ltFromBelow {t} {v} pf)

compare-lt-below : ∀ {t v} → t < v → compare t v ≡ below
compare-lt-below z<s = refl
compare-lt-below (s<s lt) = compare-lt-below lt

enforce : (threshold value : Nat) → VoxelGuard threshold value
enforce threshold value with compare threshold value
... | below = ascend z<s
... | equal = pivot refl
... | above = stay (ltFromCompare {threshold} {value} refl)

enforce-ascended-if : ∀ t v → t < v → state (enforce t v) ≡ ascended
enforce-ascended-if t v lt rewrite compare-lt-below lt = refl

enforce-ascended-only-if : ∀ t v → state (enforce t v) ≡ ascended → t < v
enforce-ascended-only-if t v with compare t v
... | below = λ _ → ltFromBelow {t} {v} refl
... | equal = λ()
... | above = λ()
