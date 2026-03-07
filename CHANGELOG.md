# Changelog

## 2026-03-07

- `26_grok_critical_scan.py` now checkpoints each completed run to `grok_critical_scan.csv` and appends per-epoch train/test trajectories to `grok_critical_scan_trajectories.csv`.
- The grokking critical scan now resumes by skipping completed `(p, weight_decay, seed)` tuples already present in the summary CSV.
- Added conservative early stopping to the grokking critical scan: a run exits only after 5 logged checkpoints in a row satisfy `test_acc >= 0.95`.
- Narrowed the active grokking onset scan to `p=97`, `seed=0`, `weight_decay ∈ {0.25, 0.30, 0.35, 0.40}`, and disabled cross-prime sanity runs for the current curve-shape pass.
- Added `26_grok_trajectory_analysis.py` to turn checkpointed grokking CSVs into milestone tables, onset-fit screens, and normalized overlay plots.
- Recorded initial grokking onset results for the narrowed scan:
  - `wd=0.25`: `t95=24980`, `final_test_acc≈0.9537`, `final_test_loss≈0.1589`
  - `wd=0.30`: `t95=21760`, `final_test_acc≈0.9545`, `final_test_loss≈0.1519`
  - `wd=0.35`: `t95=19500`, `final_test_acc≈0.9545`, `final_test_loss≈0.1478`

## 2026-02-25

- `30_delta_cone_signature_diagnose.py` now falls back to `iter` when `step` is missing, with a warning, to support embeddings that use `iter` as the step column.
