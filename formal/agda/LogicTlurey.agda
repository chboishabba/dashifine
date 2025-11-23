module LogicTlurey where

open import Agda.Builtin.Equality
open import Agda.Builtin.List
open import Agda.Builtin.Nat

------------------------------------------------------------------------
-- Local equality utilities (no stdlib needed)
------------------------------------------------------------------------

sym : ∀ {A : Set} {x y : A} → x ≡ y → y ≡ x
sym refl = refl

cong : ∀ {A B : Set} {x y : A} (f : A → B) → x ≡ y → f x ≡ f y
cong f refl = refl

open import Base369

------------------------------------------------------------------------
-- Dialectical stages
------------------------------------------------------------------------

data Stage : Set where
  seed      : Stage
  counter   : Stage
  resonance : Stage
  overflow  : Stage

next : Stage → Stage
next seed      = counter
next counter   = resonance
next resonance = overflow
next overflow  = seed

------------------------------------------------------------------------
-- Tlurey traces
------------------------------------------------------------------------

StageTrace : Nat → Stage → List Stage
StageTrace zero    _ = []
StageTrace (suc n) s = s ∷ StageTrace n (next s)

length : ∀ {A} → List A → Nat
length []       = zero
length (_ ∷ xs) = suc (length xs)

_++_ : ∀ {A} → List A → List A → List A
[]       ++ ys = ys
(x ∷ xs) ++ ys = x ∷ (xs ++ ys)

StageTrace-length : ∀ n s → length (StageTrace n s) ≡ n
StageTrace-length zero    _ = refl
StageTrace-length (suc n) s = cong suc (StageTrace-length n (next s))

next⁴ : ∀ s → spin 4 next s ≡ s
next⁴ seed      = refl
next⁴ counter   = refl
next⁴ resonance = refl
next⁴ overflow  = refl

spin-next-succ : ∀ n s → spin n next (next s) ≡ spin (suc n) next s
spin-next-succ zero    _ = refl
spin-next-succ (suc n) s = cong next (spin-next-succ n s)

StageTrace-periodic : ∀ n s → StageTrace (n + 4) s ≡ StageTrace n s ++ StageTrace 4 (spin n next s)
StageTrace-periodic zero    s = refl
StageTrace-periodic (suc n) s rewrite sym (spin-next-succ n s) =
  cong (s ∷_) (StageTrace-periodic n (next s))

StageTrace-cycle : ∀ n → StageTrace (n + 4) seed ≡ StageTrace n seed ++ StageTrace 4 seed
StageTrace-cycle n rewrite next⁴ seed = StageTrace-periodic n seed

------------------------------------------------------------------------
-- Semantics via triadic values
------------------------------------------------------------------------

stageTone : Stage → TriTruth
stageTone seed      = tri-low
stageTone counter   = tri-mid
stageTone resonance = tri-high
stageTone overflow  = tri-low

combineStage : Stage → Stage → TriTruth
combineStage a b = triXor (stageTone a) (stageTone b)

stageTone-next : ∀ s → stageTone (next s) ≡ rotateTri (stageTone s)
stageTone-next seed      = refl
stageTone-next counter   = refl
stageTone-next resonance = refl
stageTone-next overflow  = refl

resonance-combine : combineStage resonance resonance ≡ tri-mid
resonance-combine = refl
