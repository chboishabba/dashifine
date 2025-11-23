module Base369 where

open import Agda.Builtin.Equality
open import Agda.Builtin.Nat

------------------------------------------------------------------------
-- Utility: repeated rotation
------------------------------------------------------------------------

spin : {A : Set} → Nat → (A → A) → A → A
spin 0       rot x = x
spin (suc n) rot x = rot (spin n rot x)

------------------------------------------------------------------------
-- Truth values
------------------------------------------------------------------------

data TriTruth : Set where
  tri-low  : TriTruth
  tri-mid  : TriTruth
  tri-high : TriTruth

tri-index : TriTruth → Nat
tri-index tri-low  = 0
tri-index tri-mid  = 1
tri-index tri-high = 2

rotateTri : TriTruth → TriTruth
rotateTri tri-low  = tri-mid
rotateTri tri-mid  = tri-high
rotateTri tri-high = tri-low

triXor : TriTruth → TriTruth → TriTruth
triXor carrier target = spin (tri-index carrier) rotateTri target

rotateTri³ : ∀ t → rotateTri (rotateTri (rotateTri t)) ≡ t
rotateTri³ tri-low  = refl
rotateTri³ tri-mid  = refl
rotateTri³ tri-high = refl

triXor-identityˡ : ∀ t → triXor tri-low t ≡ t
triXor-identityˡ _ = refl

triXor-assoc : ∀ a b c → triXor a (triXor b c) ≡ triXor (triXor a b) c
triXor-assoc tri-low  _        _ = refl
triXor-assoc tri-mid  tri-low  c = refl
triXor-assoc tri-mid  tri-mid  c = refl
triXor-assoc tri-mid  tri-high c rewrite rotateTri³ c = refl
triXor-assoc tri-high tri-low  c rewrite rotateTri³ c = refl
triXor-assoc tri-high tri-mid  c rewrite rotateTri³ c = refl
triXor-assoc tri-high tri-high c rewrite rotateTri³ c = refl

------------------------------------------------------------------------

-- A hexadic universe: six “beats” that wrap around.

data HexTruth : Set where
  hex-0 hex-1 hex-2 hex-3 hex-4 hex-5 : HexTruth

hex-index : HexTruth → Nat
hex-index hex-0 = 0
hex-index hex-1 = 1
hex-index hex-2 = 2
hex-index hex-3 = 3
hex-index hex-4 = 4
hex-index hex-5 = 5

rotateHex : HexTruth → HexTruth
rotateHex hex-0 = hex-1
rotateHex hex-1 = hex-2
rotateHex hex-2 = hex-3
rotateHex hex-3 = hex-4
rotateHex hex-4 = hex-5
rotateHex hex-5 = hex-0

hexXor : HexTruth → HexTruth → HexTruth
hexXor carrier target = spin (hex-index carrier) rotateHex target

rotateHex⁶ : ∀ h → spin 6 rotateHex h ≡ h
rotateHex⁶ hex-0 = refl
rotateHex⁶ hex-1 = refl
rotateHex⁶ hex-2 = refl
rotateHex⁶ hex-3 = refl
rotateHex⁶ hex-4 = refl
rotateHex⁶ hex-5 = refl

hexXor-identityˡ : ∀ h → hexXor hex-0 h ≡ h
hexXor-identityˡ _ = refl

------------------------------------------------------------------------

-- A nonary universe: nine “voxels” in a ring.

data NonaryTruth : Set where
  non-0 non-1 non-2 non-3 non-4 non-5 non-6 non-7 non-8 : NonaryTruth

nonary-index : NonaryTruth → Nat
nonary-index non-0 = 0
nonary-index non-1 = 1
nonary-index non-2 = 2
nonary-index non-3 = 3
nonary-index non-4 = 4
nonary-index non-5 = 5
nonary-index non-6 = 6
nonary-index non-7 = 7
nonary-index non-8 = 8

rotateNonary : NonaryTruth → NonaryTruth
rotateNonary non-0 = non-1
rotateNonary non-1 = non-2
rotateNonary non-2 = non-3
rotateNonary non-3 = non-4
rotateNonary non-4 = non-5
rotateNonary non-5 = non-6
rotateNonary non-6 = non-7
rotateNonary non-7 = non-8
rotateNonary non-8 = non-0

nonaryXor : NonaryTruth → NonaryTruth → NonaryTruth
nonaryXor carrier target = spin (nonary-index carrier) rotateNonary target

rotateNonary⁹ : ∀ n → spin 9 rotateNonary n ≡ n
rotateNonary⁹ non-0 = refl
rotateNonary⁹ non-1 = refl
rotateNonary⁹ non-2 = refl
rotateNonary⁹ non-3 = refl
rotateNonary⁹ non-4 = refl
rotateNonary⁹ non-5 = refl
rotateNonary⁹ non-6 = refl
rotateNonary⁹ non-7 = refl
rotateNonary⁹ non-8 = refl

nonaryXor-identityˡ : ∀ n → nonaryXor non-0 n ≡ n
nonaryXor-identityˡ _ = refl
