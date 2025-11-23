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
-- Threshold guards with embedded proofs
------------------------------------------------------------------------

-- Each constructor carries the witness required to justify the
-- classification relative to the threshold.
data VoxelGuard (threshold value : Nat) : Set where
  stay   : value ≺ threshold → VoxelGuard threshold value
  pivot  : threshold ≡ value → VoxelGuard threshold value
  ascend : threshold ≺ value → VoxelGuard threshold value

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
enforce zero      zero      = pivot refl
enforce zero      (suc v)   = ascend z≺s
enforce (suc t)   zero      = stay z≺s
enforce (suc t)   (suc v) with enforce t v
... | stay p   = stay (s≺s p)
... | pivot p  = pivot (cong suc p)
... | ascend p = ascend (s≺s p)
