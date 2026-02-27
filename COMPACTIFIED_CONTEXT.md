# Compactified Context

## 2026-02-25

- HEPData delta-cone screen: best indefinite diagonal mask is `(+,-,-)` on `[v_pnorm, v_dnorm, v_arrow]`.
- Forward-step cone fraction is 59/60; only violation is `pTll_76_106` at iter 9 → 10.
- The violating step has large simultaneous drops in `v_pnorm` and `v_arrow`, with `v_dnorm` effectively null in Δ-space.
- `30_delta_cone_signature_diagnose.py` now falls back to `iter` when `step` is missing.

## 2026-02-26

- Added planning docs under `.planning/` for scale-robustness work (PROJECT/ROADMAP/STATE + 01-01 plan).
- Implemented `33_scale_robustness.py` to report pos_scale interval lengths where cone_frac_min >= threshold.
- Robustness script enforces indefinite masks by default and supports perturbation/arrow variants.
- Added forward selection modes (`order`, `arrow_rank_freeze`) and two-sided checks (`complement`, `reverse_edges`, `bidirectional`) to robustness script.
- Ran order-only and rank-freeze forward modes: interval stayed [0.10, 0.20] at threshold 1.0.
- Two-sided complement tests yielded no valid interval with v_depth (no backward steps) and with v_arrow rank-freeze (only 3 labels had both forward/backward steps).
- Two-sided reverse_edges produced no valid interval even with eps up to 1e-4.
- Two-sided bidirectional with v_arrow rank-freeze expanded with eps: [0.10,0.20] at 1e-12..1e-8; [0.10,0.30] at 1e-6; [0.10,0.85] at 1e-4.
- Added MDL-based forward mode in scale robustness script (uses E_MDL_proxy or fallback -log chi2_dof from per_label_timeseries).
- Added MDL quantile forward mode (tercile split) with explicit backward mask for two-sided coverage.
- Added flag to force MDL fallback-only (ignore E_MDL_proxy).
- MDL tercile + bidirectional sweep yields interval [0.10, 0.20]; tightest label is ptll_76_106_table (ttbar next).
- MDL fallback-only (pure -log1p chi2_dof) tercile + bidirectional also yields [0.10, 0.20], pinned by ptll_76_106_table.
- MDL fallback-only bidirectional quantiles q=0.25 and q=0.40 both keep [0.10, 0.20], pinned by ptll_76_106_table.
- Dense pos_scale scan (MDL fallback-only, q=1/3, bidirectional) shows first overall failure at 0.21; ptll_76_106_table fails first at steps 7→8 and 8→9 (forward).
- Fine scan (0.20..0.22 step 0.002) shows first failure at 0.204; single failing step ptll_76_106_table forward 8→9.
- MDL-based forward sweep (E_MDL_proxy, fallback -log1p chi2_dof) yields interval [0.10, 0.20]; pinned by ptll_76_106_table and ttbar_mtt_8tev_cms.
