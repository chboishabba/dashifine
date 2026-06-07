# Dashifine ↔ TextGraphs Bridge (ZKP framing)

## Canonical State
- `BridgeState`: `{ id, label, ternary:{self,norm,mirror}, invariants:{R,z}, phase, t_index, prev_id, next_id }`
- Source of truth for the path: `dashifine/conversational_path_plot.py` (A→I waypoints, resonance metrics).

## Encoders (C1–C3)
1) Label encoder: tokens = waypoint labels (`A`, `B`, …) or signed triples.
2) Factor encoder: tokens = factors per waypoint (`self:+`, `norm:-`, `mirror:0`, optionally `R:<sign>`, `z:<sign>`).
3) Transition encoder: tokens/edges = `prev->next`, deltas per axis, boundary flags.

## Observables (Lattice)
- Dashifine: path length, recurrence, `R`, `z`, phase boundaries, crossings.
- TextGraphs: density, SCC count/size, mean betweenness/closeness/eigenvector, Erdős–Rényi ratios, optional windowed props.
- Shared observable lattice: per-sequence or per-window vectors of the above for comparison.

## Objectives (F)
Minimize bridge cost: `ReconLoss + OrderLoss + InvariantLoss + ComplexityCost`.
- ReconLoss: can we recover the original waypoint order from the graph/stream?
- OrderLoss: temporal blur introduced by the graph construction.
- InvariantLoss: mismatch between Dashifine invariants and graph observables.
- ComplexityCost: MDL-style cost of the encoder/graph.

## Phased Plan (P)
1) Baseline: serialize A→I labels; build TextGraphs `naive_graph`; compute `graph_props`; verify exact sequence recovery.
2) Weighted: add `R`/`z` as edge/node weights; repeat metrics; compare against Dashifine table.
3) Windowed dynamics: slide over the sequence; compare local Dashifine curvature/`R` swings vs local graph density/SCC/centrality.
4) Embedding: embed serialized tokens; build weighted complete/target graph; check proximity vs ternary adjacency.
5) Comparison report: emit a compact artifact that reads the Phase 1/2 CSVs and summarizes correspondences/mismatches.
6) Variant sweep: test a small fixed family of non-local edge rules in one batch and rank them against the baseline.

## Canonical Serializer / Outputs (for Phase 1)
- Input: `BridgeState` rows emitted from the Dashifine path generator.
- Token format (baseline): `LABEL:self,norm,mirror` e.g. `A:+,+,+`.
- Artifacts to emit:
  - `dashifine_path_rows.csv` with columns `t,label,self,norm,mirror,R,z,phase`.
  - `dashifine_token_stream.txt` as a single-line space-separated stream of the canonical tokens.
  - (Optional) plain-label stream kept out of the baseline to avoid ambiguity.
- TextGraphs Phase 1 output target:
  - `textgraphs_graph_props.csv` produced by a Julia bridge that reads the token stream, builds a graph via `build_labelled_graph`, and runs `graph_props`.
- Augmented and sweep outputs:
  - `textgraphs_graph_props_augmented.csv` for the current chosen non-local rule.
  - `textgraphs_variant_props.csv` for the batched non-local rule sweep.
- Comparison outputs:
  - `bridge_reference_snapshot.json` to lock the current baseline/augmented/correlation state.
  - `bridge_variant_report.json` and `bridge_variant_report.csv` to compare baseline vs augmented and rank the current sweep variants.
- Current state:
  - baseline bridge is complete and reproducible;
  - augmented graph run is fixed and writes `textgraphs_graph_props_augmented.csv`;
  - current sweep winner is `ternary_l1_le_2`, but this is selected by a simple global graph-lift score rather than a window-aware semantic objective.
- Julia bridge behavior:
  - use a local TextGraphs-compatible implementation of the small graph-building and graph-property subset we need, so Phase 1 is reproducible even if the full `TextGraphs.jl` package has dependency/precompile issues.

## Gaps (G)
- Pick and fix one canonical serialization (C1 vs C2 vs C3) for reproducibility.
- Add conformance harness: golden outputs for A→I path (adjacency, metrics) to detect drift.
- Add MDL-style scoring across encoders.

## TODOs
- [x] Implement Phase 1 emitter: produce canonical path rows + token stream from Dashifine source.
- [x] Build TextGraphs `naive_graph` from the emitted stream and log `graph_props` alongside Dashifine metrics.
- [x] Emit a compact comparison report for Phase 1 outputs (`dashifine_path_rows.csv` vs `textgraphs_graph_props.csv`).
- [x] Add weighted Phase 2 bridge artifacts derived from `R` and `z`.
- [x] Add sliding-window Phase 2 report over the path sequence.
- [x] Lock the current bridge baseline in a compact snapshot artifact.
- [x] Add a batched non-local edge-rule sweep and score the variants against the baseline.
- [x] Emit a compact ranking report and pick the current best bridge rule.
- [ ] Add a conformance golden artifact for the Julia bridge outputs.
- [ ] Add conformance test artifact (golden JSON/CSV) for A→I path.
- [ ] Score sweep variants against windowed Dashifine behavior, not only global graph lift.
- [ ] Revisit the selected rule once the scoring includes semantic/window alignment.
- [ ] Wire MDL-style bridge cost computation (even simple length + entropy) to rank encoders.
