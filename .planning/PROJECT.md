# Dashifine Cone Robustness

## What This Is

This repository analyzes HEPData-derived embeddings to test cone-monotone dynamics in closure embeddings. The current focus is adding scale-robustness reporting and perturbation/arrow robustness checks so the “physics closure” framing is testable and repeatable.

## Core Value

Make cone-compatibility results defensible by reporting scale-robust intervals under nontriviality constraints.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Report pos_scale interval length where cone_frac_min >= threshold (per label and overall).
- [ ] Enforce nontriviality: negative-definite masks cannot “pass” in closure reports.
- [ ] Add perturbation and arrow-robustness sweeps to test stability of critical pos_scale.

### Out of Scope

- Inferring physical causality beyond the tested embedding — requires separate theory mapping.
- Changing embedding definitions or regenerating HEPData sources.

## Context

Recent runs show a stable indefinite cone with a frozen pos_scale ≈ 0.2034 and a sharp failure beyond ~0.25. The next step is to quantify scale robustness and perturbation sensitivity rather than reporting a best-fit pos_scale.

## Constraints

- **Workflow**: Planning-first per GSD.
- **Tools**: Do not install dependencies during this task.
- **Compatibility**: Avoid breaking existing CLI usage in 31/32 scripts.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Add a new scale-robustness sweep script instead of changing 31’s defaults | Keep existing CLI stable while adding new metrics | ✓ Good |

---
*Last updated: 2026-02-26 after planning “scale robustness” milestone.*
