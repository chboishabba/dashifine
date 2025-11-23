module Overflow where

open import Agda.Builtin.Equality
open import Agda.Builtin.Nat using (Nat ; zero ; suc)

------------------------------------------------------------------------
-- Local congruence lemma (no stdlib needed)
------------------------------------------------------------------------

cong : ∀ {A B : Set} {x y : A} (f : A → B) → x ≡ y → f x ≡ f y
cong f refl = refl

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
compare-≺→below z≺s      = refl
compare-≺→below (s≺s p)  = compare-≺→below p

compare-roundtrip-below : ∀ {t v} (p : t ≺ v) → compare-below→≺ (compare-≺→below p) ≡ p
compare-roundtrip-below z≺s      = refl
compare-roundtrip-below (s≺s p)  = cong s≺s (compare-roundtrip-below p)

------------------------------------------------------------------------
-- Enforcement by comparison
------------------------------------------------------------------------

enforce : (threshold value : Nat) → VoxelGuard threshold value
enforce threshold value with compare threshold value
... | below = ascend (compare-below→≺ refl)
... | equal = pivot (compare-equal→≡ refl)
... | above = stay  (compare-above→≺ refl)

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

------------------------------------------------------------------------
-- Extra comparison lemmas (if you still want them)
------------------------------------------------------------------------

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
