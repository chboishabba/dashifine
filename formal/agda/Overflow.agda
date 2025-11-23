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
-- Threshold guards with explicit proofs
------------------------------------------------------------------------

-- Each constructor records the evidence for how a measured `value`
-- relates to the `threshold`, ensuring downstream consumers cannot
-- forget the comparison witness.
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
data Order : Set where below equal above : Order

compare : Nat → Nat → Order
compare zero    zero    = equal
compare zero    (suc _) = below
compare (suc _) zero    = above
compare (suc a) (suc b) = compare a b

------------------------------------------------------------------------
-- Relations exposed by comparison tokens
------------------------------------------------------------------------

compare-below→≺ : ∀ {t v} → compare t v ≡ below → t ≺ v
compare-below→≺ {zero}   {zero}   ()
compare-below→≺ {zero}   {suc _}  refl = z≺s
compare-below→≺ {suc _}  {zero}   ()
compare-below→≺ {suc t}  {suc v}  pr   = s≺s (compare-below→≺ {t} {v} pr)

compare-above→≺ : ∀ {t v} → compare t v ≡ above → v ≺ t
compare-above→≺ {zero}   {zero}   ()
compare-above→≺ {zero}   {suc _}  ()
compare-above→≺ {suc _}  {zero}   refl = z≺s
compare-above→≺ {suc t}  {suc v}  pr   = s≺s (compare-above→≺ {t} {v} pr)

compare-equal→≡ : ∀ {t v} → compare t v ≡ equal → t ≡ v
compare-equal→≡ {zero}  {zero}  refl = refl
compare-equal→≡ {zero}  {suc _} ()
compare-equal→≡ {suc _} {zero} ()
compare-equal→≡ {suc t} {suc v} pr = cong suc (compare-equal→≡ {t} {v} pr)

compare-≺→below : ∀ {t v} → t ≺ v → compare t v ≡ below
compare-≺→below z≺s = refl
compare-≺→below (s≺s p) = compare-≺→below p

compare-roundtrip-below : ∀ {t v} (p : t ≺ v) → compare-below→≺ (compare-≺→below p) ≡ p
compare-roundtrip-below z≺s = refl
compare-roundtrip-below (s≺s p) = cong s≺s (compare-roundtrip-below p)

enforce : (threshold value : Nat) → VoxelGuard threshold value
enforce threshold value with compare threshold value
... | below = ascend (compare-below→≺ refl)
... | equal = pivot (compare-equal→≡ refl)
... | above = stay (compare-above→≺ refl)

------------------------------------------------------------------------
-- Correctness of enforcement
------------------------------------------------------------------------

enforce-ascended-if : ∀ {t v} (p : t ≺ v) → enforce t v ≡ ascend p
enforce-ascended-if {t} {v} p with compare t v | compare-≺→below p
... | .below | refl = cong ascend (compare-roundtrip-below p)

only-if : ∀ {t v} → state (enforce t v) ≡ ascended → t ≺ v
only-if {t} {v} with enforce t v
... | stay   _ = λ ()
... | pivot  _ = λ ()
... | ascend p = λ _ → p
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
