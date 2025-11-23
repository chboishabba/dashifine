import Lake
open Lake DSL

package «FormalCore»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.9.0"

@[default_target]
lean_lib «FormalCore»
