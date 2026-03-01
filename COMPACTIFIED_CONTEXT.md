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
- Added snap/exception filtering based on beta-norm quantiles or absolute thresholds in scale robustness script.
- Added snap signature filter (zeroing count + chi2 spike + MDL descent).
- Added joint snap signature (beta quantile + |Δdnorm| quantile + chi2 spike + MDL descent + zeroing).
- Added shrink-ratio snap signature (component shrink by kappa + chi2 spike + MDL descent).
- Added shrink+dnorm joint snap signature (shrink_count >= N and |Δdnorm| in top quantile).
- Snap-filter (beta_norm_abs=1.0) extends window: coarse [0.10,0.22], fine [0.20,0.224], still pinned by ptll_76_106_table.
- Snap-signature trials: sig2 (beta q=0.90, chi2>=1.1) keeps [0.10,0.20]; sig3 (beta q=0.85, chi2>=1.02) expands to [0.10,0.22], still pinned by ptll.
- Joint signature (beta q=0.85, |Δdnorm| q=0.85, chi2>=1.02, zero_min=0) expands to [0.10,0.22], still pinned by ptll.
- Shrink-ratio signature (kappa=0.1): shrink_min=2 no change; shrink_min=1 expands to [0.10,0.22], ptll still pins.
- Shrink+dnorm joint signature (kappa=0.5, shrink_min=1, |Δdnorm| q=0.85) expands to [0.10,0.22], ptll still pins.
- Shrink+dnorm variants (q=0.70 or kappa=0.2) still yield [0.10,0.22].
- Added shrink+dnorm OR chi2 snap signature option.
- Added min-steps-per-label guard to exclude mode-starved labels.
- Added 34_snap_sweep.py; quota-preserving snap sweep shows full coverage with keep_frac>=0.3 and window expands to [0.10,0.22] for keep_frac<=0.7.
- Snap score invariance: beta_norm vs beta_norm_chi2 yields identical loss curve and pinned label; comparison in `snap_sweep_q40_min3_score_comparison.csv`.
- Baseline coverage with snap disabled (MDL fallback q=0.40, bidirectional, min-steps=3): uniform n_fwd=5, n_bwd=5 per label; interval [0.10, 0.20]; outputs `scale_mdl_fallback_q40_bidirectional_nosnap_min3_summary.csv` and `_coverage.csv`.
- Arrow/shape independence check: Pearson r ≈ -9.76e-4, Spearman r ≈ -0.1008; outputs `arrow_shape_independence.csv`, `arrow_shape_independence.png`, `arrow_shape_independence_summary.txt`.
- Ultrametric triangle check (LCP-depth metric on ternary lens vectors): 0 violations across 40,000 trials; output `ultrametric_triangle_report.csv`.
- shrink_or_chi2: with chi2>=1.05 -> [0.10,0.22] (ptll still pins); with chi2>=1.01 -> [0.10,0.30], ptll removed, new pin z_pt_7tev_atlas.
- shrink_or_chi2 with chi2>=1.015 and dnorm q=0.90 keeps [0.10,0.22], ptll still pins (10→11 not caught).
- shrink_or_chi2 with chi2>=1.012 and min-steps-per-label=5 drops all labels (over-filtered).
- MDL tercile + bidirectional sweep yields interval [0.10, 0.20]; tightest label is ptll_76_106_table (ttbar next).
- MDL fallback-only (pure -log1p chi2_dof) tercile + bidirectional also yields [0.10, 0.20], pinned by ptll_76_106_table.
- MDL fallback-only bidirectional quantiles q=0.25 and q=0.40 both keep [0.10, 0.20], pinned by ptll_76_106_table.
- Dense pos_scale scan (MDL fallback-only, q=1/3, bidirectional) shows first overall failure at 0.21; ptll_76_106_table fails first at steps 7→8 and 8→9 (forward).
- Fine scan (0.20..0.22 step 0.002) shows first failure at 0.204; single failing step ptll_76_106_table forward 8→9.
- Critical-scale report: min s*=(Δp^2+Δa^2)/Δd^2 = 0.2034378752 at ptll_76_106_table 8→9; second smallest 0.2099217488 at 7→8. Saved `scale_mdl_fallback_q33_bidirectional_s_star.csv`.
- ptll_76_106_table forward r* list: r1=0.2034379 (8→9), r2=0.2099217 (7→8), r3=0.2203354 (9→10), r4=0.2249192 (10→11); gap ratio r2/r1 ≈ 1.0319. Outputs: `ptll_76_106_table_r_star_forward.csv`, `ptll_76_106_table_r_star_forward.png`.
- Whitening comparison: diag-whitened r* cluster persists with scaled values (~0.0276..0.0306); full-whitened cluster persists (~0.0847..0.1136) with ordering slightly changed. Output: `ptll_76_106_table_r_star_forward_whiten.csv`.
- Canonical normalization (full-whitened, MDL fallback q=1/3, bidirectional): using Q_p99=0 implies s = quantile_0.01(sn/sp)=0.0887006687; overall min/mean 0.875/0.9904; holdout (exclude ptll) min/mean 1.0. Outputs: `full_whiten_qpin_summary_fixed.csv`, `full_whiten_qpin_per_label_fixed.csv`.
- MDL-based forward sweep (E_MDL_proxy, fallback -log1p chi2_dof) yields interval [0.10, 0.20]; pinned by ptll_76_106_table and ttbar_mtt_8tev_cms.
