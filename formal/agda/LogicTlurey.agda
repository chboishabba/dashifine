module formal.agda.LogicTlurey where

open import Agda.Builtin.List
open import Agda.Builtin.Nat

open import formal.agda.Base369

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
StageTrace zero    s = s ∷ []
StageTrace (suc n) s = s ∷ StageTrace n (next s)

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
