module Base369 where

open import Agda.Builtin.Nat

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

------------------------------------------------------------------------
-- Utility: repeated rotation
------------------------------------------------------------------------

spin : {A : Set} → Nat → (A → A) → A → A
spin 0       rot x = x
spin (suc n) rot x = rot (spin n rot x)
