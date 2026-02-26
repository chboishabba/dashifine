# TODO (Dashifine / HEPData cone screen)

- Test log (2026-02-26):
  - Data acquisition / prep:
    - Added HEPData table support for record ids (e.g., ins1663452) + table names, and for covariance headers "Matrix element".
    - Downloaded/processed bundles to `hepdata_npz_all` and lens outputs to `hepdata_to_dashi_all`.
    - Ran DASHI-native contraction on all tables -> `hepdata_dashi_native/*_dashi_native_metrics.csv`.
    - Built timeseries -> `hepdata_lyapunov_test_out_all/per_label_timeseries.csv`.
    - Built closure embedding -> `hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv`.

  - Cone screens (3D, x = [v_pnorm, v_dnorm, v_arrow], arrow = v_depth, indefinite only):
    - Combined labels (13 labels, 156 forward steps):
      - Unweighted best: mask `-1,1,-1`, cone_frac_min=0.5833, cone_frac_mean=0.8846, max_Qd=5.6057.
      - Weighted best: mask `-1,1,-1`, pos_scale≈0.2034, cone_frac_min=1.0, max_Qd≈0.
      - Filtered+weighted: pos_scale≈0.2099, cone_frac_min=1.0.

  - Freeze+transfer (no search, no refit, no filtering):
    - Frozen mask `-1,1,-1`, pos_scale=0.2034:
      - All labels: cone_frac_min=1.0, cone_frac_mean=1.0, max_Qd≈0.
      - Excluding `ptll_76_106_table`: cone_frac_min=1.0, cone_frac_mean=1.0, max_Qd≈0.
    - Per-label margins: strictly negative (tiny), confirming no ε-hugging.

  - Q-margin distribution (frozen mask/scale):
    - Median Q ≈ -0.00318.
    - Near-null fractions: |Q|<=1e-6: ~0.224; |Q|<=1e-2: ~0.564.
    - Plots: `cone_margin_hist.png`, `cone_margin_ecdf.png`, `cone_margin_per_label_median.png`.

  - pos_scale sweep (frozen mask `-1,1,-1`, no filtering):
    - Perfect closure for pos_scale ∈ [0.10, 0.20].
    - Hard break at pos_scale ≈ 0.25 with cone_frac_min dropping to 0.5833.
    - Monotone increase in max_Q with pos_scale up to ~5.6 at pos_scale=1.0.

  - Whitening / normalization diagnostics:
    - v_dnorm is near-null in MAD but not in std (spiky axis); MAD normalization destabilizes.
    - 2D tests (dropping v_dnorm) show unweighted cone weak; weighted requires small pos_scale.
    - Global whitening did not move critical scale toward 1.0; closure degraded.

- HEPData sources (table set change in scripts):
  - removed (previous set)
  - - (129890, 129891) pTll_50_76
  - - (129892, 129893) pTll_76_106
  - - (129894, 129895) pTll_106_170
  - - (129896, 129897) pTll_170_350
  - - (129902, 129903) phistar_50_76
  - added (current set uses table JSON URLs; some are central+cov pairs)
  - + Z/gamma* pT (ATLAS 7 TeV): https://www.hepdata.net/download/table/ins1300647/Table 1/json
  - + ttbar m_tt (CMS 8 TeV): https://www.hepdata.net/download/table/ins1370682/Table 39/json
  - + ttbar m_tt covariance (CMS 8 TeV): https://www.hepdata.net/download/table/ins1370682/Table 40/json
  - + H->gamma gamma pT (ATLAS 8 TeV): https://www.hepdata.net/download/table/ins1391147/Table 2/json
  - + Dijet angular chi (CMS 7 TeV): https://www.hepdata.net/download/table/ins889175/Table 1/json
- HEPData URL patterns:
  - record JSON: https://www.hepdata.net/record/<record_id>?format=json
  - data JSON (example): https://www.hepdata.net/record/data/82308/322783/1/
  - note: the JSON button on record pages uses `?format=json` and may redirect to a `record/data/...` JSON endpoint.

- Decide whether to filter single-step jump events (e.g., percentile threshold on ||Δx||) before cone screening.
- Evaluate a weighted diagonal Lorentz form (fit scale weights) to test if the single violation is a coordinate-scaling artifact.
- Add an option to auto-detect near-null Δ-axes (variance threshold) and drop them from the signature scan.
- Add a short note in the cone-screen CLI help explaining the expected step column (`step` vs `iter`).
