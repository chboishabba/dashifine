# Changelog

## 2026-03-07

- `26_grok_critical_scan.py` now checkpoints each completed run to `grok_critical_scan.csv` and appends per-epoch train/test trajectories to `grok_critical_scan_trajectories.csv`.
- The grokking critical scan now resumes by skipping completed `(p, weight_decay, seed)` tuples already present in the summary CSV.
- Added conservative early stopping to the grokking critical scan: a run exits only after 5 logged checkpoints in a row satisfy `test_acc >= 0.95`.
- Narrowed the active grokking onset scan to `p=97`, `seed=0`, `weight_decay ∈ {0.25, 0.30, 0.35, 0.40}`, and disabled cross-prime sanity runs for the current curve-shape pass.
- Added `26_grok_trajectory_analysis.py` to turn checkpointed grokking CSVs into milestone tables, onset-fit screens, and normalized overlay plots.
- `26_grok_trajectory_analysis.py` now accepts multiple summary/trajectory CSV inputs so coarse and refinement scans can be analyzed together without manual merging.
- `26_grok_trajectory_analysis.py` now also fits a shared Gompertz candidate and a logistic baseline to the `epoch / t50` normalized test-accuracy curves, writing `grok_gompertz_fit.csv` and `grok_gompertz_norm_t50.png`.
- The first normalized shape-law screen on the combined 7-point dataset favored the simple logistic baseline over the shared-parameter Gompertz candidate, so the project claim remains the time-rescaled trajectory family rather than a settled Gompertz law.
- Recorded the current mechanistic interpretation in the grokking theorem note as deterministic metastable escape from a memorization regime, with a sigmoid-like post-escape rise as the next shape-law target.
- `26_grok_trajectory_analysis.py` now also fits a rising-phase-only logistic law with a per-run onset shift, writing `grok_rise_logistic_fit.csv` and `grok_rise_logistic_norm_t50.png`.
- The first rising-phase-only screen on the combined 7-point dataset gave `mse≈0.00119`, which supports the current “metastable plateau + shared sigmoid-like post-escape rise” interpretation without upgrading the full-trajectory claim to a single closed-form law.
- `26_grok_trajectory_analysis.py` now also fits the same rising-phase logistic law to normalized post-`t10` test-loss progress, writing `grok_rise_loss_logistic_fit.csv` and `grok_rise_loss_logistic_norm_t50.png`.
- The first loss-side rising-phase screen gave `mse≈0.00197`, so the smoother loss proxy did not improve the shared post-escape law on the current 7-point dataset.
- `26_grok_trajectory_analysis.py` now also fits a shared post-escape logistic law with a learned per-run onset shift, writing `grok_rise_logistic_fitted_t0_fit.csv` and `grok_rise_logistic_fitted_t0_norm_t50.png`.
- The fitted-onset screen improved the shared rise fit to `mse≈0.000351`, far better than the fixed-`t10` rise fit and essentially matching the best full normalized logistic baseline.
- `26_grok_trajectory_analysis.py` now also compares simpler onset choices for the shared post-escape logistic law, including fixed `t20` and curvature-derived onset shifts.
- On the current 7-point dataset, fixed `t20` improved the rise fit to `mse≈0.000714`, while the naive curvature onset was poor (`mse≈0.01474`), so the fitted-onset benchmark remains the strongest version.
- `26_grok_trajectory_analysis.py` now also fits a shared normalized onset law `t0 = c * t50`, writing `grok_rise_logistic_fixed_ct50_fit.csv` and `grok_rise_logistic_fixed_ct50_norm_t50.png`.
- The shared-onset screen achieved `mse≈0.000360` with `c≈0.8055`, essentially matching the per-run fitted-onset result and giving the cleanest current empirical law for the rise phase.
- Updated `GROKKING_TIME_RESCALING_NOTE.md` and the README grokking section so the headline result now states the shared-onset logistic law directly instead of framing Gompertz as the next primary target.
- Added `26_grok_critical_scan_refine.py` for the lower-`weight_decay` follow-up band, writing refinement outputs to `2_grok_critical_scan_refine.csv` and `2_grok_critical_scan_refine_trajectories.csv`.
- Recorded initial grokking onset results for the narrowed scan:
  - `wd=0.25`: `t95=24980`, `final_test_acc≈0.9537`, `final_test_loss≈0.1589`
  - `wd=0.30`: `t95=21760`, `final_test_acc≈0.9545`, `final_test_loss≈0.1519`
  - `wd=0.35`: `t95=19500`, `final_test_acc≈0.9545`, `final_test_loss≈0.1478`
- Recorded lower-band refinement results:
  - `wd=0.20`: `t95=30780`, `final_test_acc≈0.9526`, `final_test_loss≈0.1508`
  - `wd=0.22`: `t95=28400`, `final_test_acc≈0.9519`, `final_test_loss≈0.1642`
  - `wd=0.24`: `t95=25900`, `final_test_acc≈0.9535`, `final_test_loss≈0.1600`
- Combined coarse + refinement analysis supports `t95 ~ 1 / weight_decay` as the best simple 2-parameter onset screen (`R²≈0.9976`) and shows stronger collapse under `epoch / t50` than `epoch / t95`.
- Added `GROKKING_TIME_RESCALING_NOTE.md` and a README pointer to it, recording the current reduced-model conjecture, the fast-slow theorem target, and the explicit limitation that the result is presently supported only in one architecture/task regime.

## 2026-02-25

- `30_delta_cone_signature_diagnose.py` now falls back to `iter` when `step` is missing, with a warning, to support embeddings that use `iter` as the step column.
