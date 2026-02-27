# Roadmap: Dashifine Cone Robustness

## Overview

Ship a robustness reporting layer for cone tests, including scale-interval scoring and perturbation/arrow stability checks, while preserving existing analysis scripts.

## Domain Expertise

None

## Phases

- [x] **Phase 1: Scale Robustness Reporting** - Add interval-based scoring and nontriviality gates.
- [ ] **Phase 2: Perturbation & Arrow Robustness** - Sweep eps/jump/noise/arrow variants and report stability.
- [ ] **Phase 3: Documentation & Results Capture** - Update TODO/summary with new outputs.

## Phase Details

### Phase 1: Scale Robustness Reporting
**Goal**: Report pos_scale interval lengths per label and overall instead of best-fit pos_scale.
**Depends on**: Nothing
**Research**: Unlikely
**Plans**: 1 plan

Plans:
- [x] 01-01: Implement robustness sweep script + outputs.

### Phase 2: Perturbation & Arrow Robustness
**Goal**: Add perturbation variants (noise, resample, eps/jump) and arrow column sweeps.
**Depends on**: Phase 1
**Research**: Unlikely
**Plans**: 1 plan

Plans:
- [ ] 02-01: Add variant runner + reporting fields.

### Phase 3: Documentation & Results Capture
**Goal**: Document new metrics/outputs and run notes.
**Depends on**: Phase 2
**Research**: Unlikely
**Plans**: 1 plan

Plans:
- [ ] 03-01: Update TODO/summary after running new sweeps.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scale Robustness Reporting | 1/1 | Complete | 2026-02-26 |
| 2. Perturbation & Arrow Robustness | 0/1 | Not started | - |
| 3. Documentation & Results Capture | 0/1 | Not started | - |
