# Agda logic sandbox

This directory contains an Agda sketch of the triadic/hexadic/nonary logic components referenced in the research notes. The goal is to make the structures explicit enough to type-check while still leaving room for future elaboration.

## Files

- `Base369.agda`: Core truth-value universes and primitive operators (rotations and XOR-like composition) for triadic, hexadic, and nonary dialects.
- `LogicTlurey.agda`: A minimal Tlurey cycle encoded as a datatype of stages with a recursive trace generator that enforces dialectical progression.
- `Overflow.agda`: A dependent guard that links numeric thresholds to voxel state escalations, forcing promotion when an overflow proof is present.

Each module is standalone and relies only on `Agda.Builtin` modules to ease compilation in minimal environments.
