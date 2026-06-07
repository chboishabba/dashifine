# Compactified Context

## 2026-03-24

- Quantum-formalism status across sibling repos:
  - `dashiQ` is now the intended home for the bridge/internalization note and
    simulator-facing formalism.
  - This repo's quantum files remain the concrete classical side:
    CHSH utilities, qutrit embeddings, SSH/lattice helpers, and
    `quantum_defect` demos under `newtest/`.
  - These should be described as quantum-faithful classical simulations /
    lattice realizations of known quantum formalisms, not as quantum hardware
    execution or quantum advantage.
  - The useful handoff to `dashiQ` is:
    - state representations already used here
    - measurement/operator patterns
    - entanglement / CHSH / qutrit embedding examples
    - current limitations from staying at the NumPy / dense-linear-algebra layer
  - Local archive cross-check (2026-03-24, source=`db`) sharpened the wording:
    - `P-adic quantum systems`
      - online UUID: `6919bf75-af7c-8324-b2be-bfc2306d8208`
      - canonical thread ID: `c5adc26f07706a65a5da6043eb91810f3041c9c0`
      - repo-facing decision: keep p-adic material in the formal/simulator lane
        for now, not as a quantum-hardware claim.
    - `Quarter turn in quantum`
      - online UUID: `690e6469-9508-8320-86b4-669fe11d6245`
      - canonical thread ID: `17bde1e6b2b7d785009b992bfaa1c4d74298dcb8`
      - repo-facing decision: the quarter-turn `J` is the main reusable bridge
        object, acting as local complex structure / phase rotation in the
        lattice-facing demos.
    - `Math Prof Outreach Stage`
      - online UUID: `69aa52b4-6f7c-839f-aa7f-d120ffe0c1ad`
      - canonical thread ID: `decf9e3cde5ccdec0c51ad8aab15999201503998`
      - repo-facing decision: wave-lift / module language is real and relevant,
        but still extension-point territory rather than finished physics
        closure.

- Dashifine ↔ TextGraphs bridge status:
  - Phase 1 is complete: the A→I conversational path is serialized into `dashifine_path_rows.csv` and `dashifine_token_stream.txt`, and the base graph metrics are emitted in `textgraphs_graph_props.csv`.
  - Phase 2 is complete on both Python and Julia sides: weighted transitions and sliding-window summaries match exactly across `bridge_phase2.json` / `bridge_phase2.csv` and the Julia outputs `textgraphs_weighted_summary.csv`, `textgraphs_weighted_edges.csv`, and `textgraphs_window_summary.csv`.
  - Current correlation read (`bridge_correlation.csv`):
    - `R_mean_dash` vs `R_mean_tg` = `1.0`
    - `z_mean_dash` vs `z_mean_tg` = `1.0`
    - `R_range_dash` vs `transition_weight_sum_tg` ≈ `0.624`
    - `z_mean_dash` vs `transition_weight_sum_tg` ≈ `-0.752`
  - Interpretation: the bridge is now conservative/invariant-preserving. TextGraphs is not just carrying the sequence; it also carries the same windowed Dashifine summaries numerically.
- Augmented/non-local graph result:
  - `bridge_textgraphs_augmented.jl` now runs successfully and writes `textgraphs_graph_props_augmented.csv`.
  - Adding non-local similarity edges changed the graph from a simple chain into a strongly connected graph:
    - density `0.1 -> 0.288888...`
    - SCC count `10 -> 1`
    - largest SCC `1 -> 10`
    - mean closeness `0.1571 -> 0.3681`
    - mean eigenvector centrality `0.1020 -> 0.2975`
  - Interpretation: this is the first point where the graph layer "lights up" structurally rather than only mirroring a path.
- Batched rule sweep result:
  - `bridge_textgraphs_sweep.jl` emits `textgraphs_variant_props.csv`.
  - `bridge_variant_report.json` / `bridge_variant_report.csv` lock the current baseline and rank a small family of non-local rules.
  - Current ranking:
    - `ternary_l1_le_2` score `1.9956`
    - `rz_l1_le_1` score `1.7869`
    - `ternary_l1_le_1` score `1.7794`
    - `hybrid_l1_recurrence` score `1.7794`
    - `state_recurrence` score `1.6606`
  - Current selected rule: `ternary_l1_le_2`
  - Important caveat: this is a simple global graph-lift score, not yet an MDL-style or window-correlation-aware selector.
- Current intended next move:
  - add a conformance/golden artifact for the Julia bridge outputs,
  - then extend selection beyond global graph lift by scoring variants against windowed Dashifine behavior (`R`, `z`, phase drift), not only density/SCC/centrality lift.

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
- Masked Orthogonal Split test (3D closure embedding): G=diag([-1,0.2034,-1]), P projects onto coords [0,1]; self_adj_error=0, cross_max_abs≈2.40e-16, energy_residual_max_abs≈6.66e-16; output `masked_orthogonal_split_report.csv` (plus `G_mask3.npy`, `P_shape01_from_G_mask3.npy`).
- Orbit profile tables regenerated with signed block permutations (Sp×Sq, sign flips), shells |Q|=1,2, in `orbit_profiles/` via `40_generate_orbit_profiles.py`. Key shell1 profiles: p1q2=[8,4,2], p2q1=[8,4,2], p3q1=[24,6,2], p4q0=[8], p2q2=[16,16,4,4], p1q3=[24,6,2], p0q4=[8]. Shell2 profiles: p1q2=[4], p2q1=[4], p3q1=[16,12], p4q0=[24], p2q2=[4,4], p1q3=[16,12], p0q4=[24].
- Integrity decision: legacy `OrbitProfileExternal` hard-coded profiles (all_code39/40) are from a different symmetry action; they must be archived/renamed. Use CSV-backed signed-block profiles as authoritative and add an arrow/orientation tag to disambiguate (3,1) vs (1,3).
- Cone monotonicity visualization generated: `viz_cone_monotone.gif` (Q(Δs) vs iteration across labels, script `41_viz_cone_monotone.py`).
- In viz_cone_monotone, one label dips to ~-40 (next best ~-30) then recovers toward 0 by iter ~12 (≈-10 at ~11); consistent with a transient deep-interior step followed by smaller steps approaching the cone boundary (cone containment only requires Q≤0).
- Closure flow visualization generated: `viz_closure_flow.gif` (PCA trajectories by label, script `42_viz_closure_flow.py`).
- Ultrametric geometry visualization generated: `viz_ultrametric_tree.gif` (MDS on LCP ultrametric distances from `lenses_ternary.csv`, script `43_viz_ultrametric_tree.py`).
- Ultrametric grid visualization generated: `viz_ultrametric_grid.gif` (2x2 panels: shell, orbit ID proxy, bin index, attractor clusters; script `44_viz_ultrametric_grid.py`).
- Additional field/density visualizations: prefix-tree density (`viz_tree_density.png`), 2D density+flow field (`viz_density_flow.png`), basin/time-to-converge (`viz_basin_time.png`) via scripts `45_viz_tree_density.py`, `46_viz_density_flow.py`, `47_viz_basin_time.py`.
- Added contraction/flow diagnostics: `viz_contraction_rate.png` with `contraction_rate.csv` (log mean pairwise distance vs iteration) and flow divergence heatmap `viz_flow_divergence.png` via scripts `48_contraction_rate.py`, `49_viz_flow_divergence.py`.
- Interpretation notes: ultrametric MDS ring/lobe structure is expected for tree metrics; closure flow shows convergence to a common endpoint (global attractor); cone plot shows Q(Δs) ≤ 0 with deep dips and recovery toward 0 (approaching boundary). Signature lock requires CSV-backed signed-block profiles and arrow/orientation to disambiguate (3,1) vs (1,3).
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

## 2026-03-07

- Grokking critical scan workflow narrowed to the `p=97` mod-multiplication task only; cross-prime sanity runs were disabled to focus on curve shape near onset.
- `26_grok_critical_scan.py` now writes checkpointed outputs after each completed run:
  - `grok_critical_scan.csv` for per-run summaries
  - `grok_critical_scan_trajectories.csv` for per-epoch train/test loss and accuracy
- The grokking scan now resumes by skipping completed `(p, weight_decay, seed)` tuples already present in `grok_critical_scan.csv`.
- Conservative early stopping was added to the grokking scan: a run stops only after 5 logged checkpoints in a row with `test_acc >= 0.95`.
- Current coarse scan configuration for onset mapping:
  - `seeds_main = [0]`
  - `wds_main = [0.25, 0.30, 0.35, 0.40]`
  - `primes_extra = []`
- Completed grokking runs so far (`p=97`, `seed=0`):
  - `wd=0.25`: `t_fit=80`, `t95=24980`, `final_test_acc≈0.9537`, `final_test_loss≈0.1589`
  - `wd=0.30`: `t_fit=80`, `t95=21760`, `final_test_acc≈0.9545`, `final_test_loss≈0.1519`
  - `wd=0.35`: `t_fit=80`, `t95=19500`, `final_test_acc≈0.9545`, `final_test_loss≈0.1478`
- Trajectory interpretation from the completed grokking runs:
  - train memorization saturates by ~epoch 80
  - test accuracy remains near chance at epoch 80
  - a long flat/chance plateau is followed by a late rapid generalization phase
  - increasing `weight_decay` shifts the onset earlier without materially changing the final thresholded accuracy in the completed cases
- Immediate analysis plan for grokking:
  - finish the coarse `wd` grid
  - fit onset times (`t50`, `t80`, `t90`, `t95`) against `weight_decay`
  - overlay normalized trajectories, especially by `epoch / t50` and `epoch / t95`
  - prefer smooth proxies like `test_loss` over raw accuracy for theorem-oriented follow-up
- Added `26_grok_trajectory_analysis.py` to automate the first-pass grokking analysis from checkpointed CSVs.
  - Outputs milestone table `grok_milestones.csv`
  - Outputs onset-fit screening table `grok_onset_fit_screen.csv`
  - Outputs raw and normalized overlay plots for `test_acc` and raw `test_loss`
  - Now accepts multiple coarse/refinement summary and trajectory CSVs so the lower-`wd` refinement band can be analyzed jointly with the coarse scan
  - Now also outputs a normalized shape-law comparison:
    - `grok_gompertz_fit.csv`
    - `grok_gompertz_norm_t50.png`
  - The shape-law screen compares a shared-parameter Gompertz candidate against a simple logistic baseline on the `epoch / t50` normalized curves
- Added `26_grok_critical_scan_refine.py` for the lower-`wd` follow-up band with a longer epoch budget and separate output files.
- Lower-band refinement results (`p=97`, `seed=0`):
  - `wd=0.20`: `t95=30780`, `final_test_acc≈0.9526`, `final_test_loss≈0.1508`
  - `wd=0.22`: `t95=28400`, `final_test_acc≈0.9519`, `final_test_loss≈0.1642`
  - `wd=0.24`: `t95=25900`, `final_test_acc≈0.9535`, `final_test_loss≈0.1600`
- Combined 7-point onset dataset (`wd ∈ {0.20, 0.22, 0.24, 0.25, 0.30, 0.35, 0.40}`):
  - `t95` decreases monotonically with `weight_decay`
  - final test accuracy stays near-constant (~0.952 to ~0.956)
  - `26_grok_trajectory_analysis.py` run on the combined dataset yields:
    - best simple onset screen: `t95 ~ 1 / wd` with `R²≈0.9976`
    - next best simple screen: `log t95 ~ 1 / wd` with `R²≈0.9961`
    - better trajectory collapse under `epoch / t50` than `epoch / t95`
- Current best empirical interpretation of the grokking data:
  - one common delayed-generalization curve family across the tested `weight_decay` band
  - a fast memorization phase followed by a slower delayed-generalization phase
  - `weight_decay` acts primarily as a time-rescaling parameter for the slow phase
  - a more specific current mechanistic hypothesis is deterministic metastable escape from a memorization regime, followed by a sigmoid-like post-escape rise
- Current shape-law status:
  - the first shared-parameter normalized-accuracy screen did not support upgrading the claim to a Gompertz law
  - on the present 7-point dataset, the simple logistic baseline fit better than the shared Gompertz candidate
  - the stronger current claim remains the time-rescaled family / fast-slow interpretation, not a fixed closed-form rise law
  - the best current shape-language is “sigmoid-like post-escape rise” rather than a settled Gompertz law
  - a stricter rising-phase-only screen was then added: fit a shared logistic rise on the post-`t10` segment with a per-run onset shift
  - that rising-phase fit achieved `mse≈0.00119`, which is consistent with the “metastable delayed plateau + shared sigmoid-like post-escape rise” picture
  - because it is evaluated on the shifted post-onset segment, it should be read as support for the post-escape law rather than as a direct replacement for the full-trajectory screens
  - the same post-`t10` logistic screen was repeated on normalized test-loss progress and achieved `mse≈0.00197`
  - on the present 7-point dataset, the loss-side proxy does not tighten the shared post-escape law; the accuracy-side rise fit is cleaner
  - replacing the fixed `t10` shift with a fitted per-run onset shift materially improved the shared post-escape logistic fit to `mse≈0.000351`
  - the learned onset shifts are nearly constant in normalized units (`t0 / t50 ≈ 0.81` across all seven runs), which suggests the onset location itself may also be part of the shared curve family
  - this is the strongest current shape result: once onset is aligned, the post-escape rise is very well described by a shared logistic law
  - lower-complexity onset tests were then compared against the fitted-`t0` benchmark:
    - fixed `t20` shift improved to `mse≈0.000714`, so it captures part but not all of the onset-alignment gain
    - naive curvature onset performed poorly (`mse≈0.01474`) and is not a good onset proxy on the present dataset
  - the natural next simplification test was then run: a single shared normalized onset location near `0.81 * t50`
  - one shared onset `t0 = c * t50` with `c≈0.8055` achieved `mse≈0.000360`, nearly identical to the per-run fitted-onset fit (`mse≈0.000351`)
  - this is the cleanest current rise-phase law: after shifting by a shared normalized onset near `0.81 * t50`, the post-escape rise is very well described by a shared logistic curve
- Current theorem target / reduced-model conjecture:
  - near-critical grokking admits a coarse fast-slow description with a one-dimensional slow variable controlling late generalization
  - empirical template: `G(t; λ) ≈ F((t - T_mem) / τ(λ))`
  - flagship conjecture: regularization primarily rescales the clock of a common delayed-generalization flow rather than changing the trajectory family
- Added `GROKKING_TIME_RESCALING_NOTE.md` as the compact theorem-note draft for the current grokking result.
  - It states the empirical observation, reduced-model conjecture, minimal assumptions, candidate theorem target, inverse-law threshold-time target, and limitation statement in one place.
  - The note deliberately treats the fast-slow / Lyapunov-style interpretation as a supported conjecture rather than an established theorem.

## 2026-03-09

- Chat archive sync completed for the user-supplied link set: 38 links collapsed to 37 unique online UUIDs; duplicate link was `695c4632-cc44-8320-96d5-e0309cf28a8a`.
- Resolution status: all 37 UUIDs now resolve in `~/chat_archive.sqlite`; 13 were already present in the archive and 24 were pulled live into the canonical DB before docs were updated.
- Active repo-facing themes recovered from the synced threads:
  - cone monotonicity / closure status: branch family around `699dc65b-6510-839a-8b31-2ea717285a10` sharpened the current need to consolidate the cone-premise diagnostics and branch outputs into one authoritative closure narrative.
  - wave/interference/kernel direction: `695c4632-cc44-8320-96d5-e0309cf28a8a`, `6993e106-93b4-83a1-930c-46ce77c1affe`, and `69984670-ff48-839a-8dee-65006e9986c9` reinforce the current benchmark line: sparse wave-field learning, interference demos, and explicit closure criteria.
  - LES / spectral implementation direction: `69718c29-6bcc-8324-b9e9-e412af8c89eb`, `6978944d-f0f8-8321-b9ca-a4e6aee51db9`, and `697c3293-9ed4-839a-ad1c-3367cdc388db` point to CPU/GPU parity, spectral-gradient implementation, and filament/fining follow-through.
  - formalism branch family: `696d865d-79a8-8321-a88c-2e37b963a4be`, `696dbd54-8818-8324-b66b-70e7bdf32d2b`, `696ece79-7478-8324-b12a-e77ee0570eec`, `696ed143-2ddc-8324-9b6e-aced5c5954d8`, and the `88923ac...` branch set remain theory context, not immediate Dashifine implementation work.
  - closure-gap diagnosis from `69aa52b4-6f7c-839f-aa7f-d120ffe0c1ad` (`Math Prof Outreach Stage`): the archive-backed verdict is “mathematical closure spine mostly there, physics program still far”; the missing pieces are still dynamics, gauge/constraint closure, matter/recovery, and a broader limit theory rather than more kinematic/signature refinements.
  - wave-lift status from the same thread: archive material says the broader DASHI program already has a wave-facing bridge (`WaveLiftEvenSubalgebra`, orbit-shell generating series, wave-lift/even-subalgebra language), but this repo currently documents the concrete Python wave/interference experiments much better than that abstract wave-lift bridge.
  - term check on `Math Prof Outreach Stage`: the `ψ` symbol itself does appear, but sparsely; the archive uses plain `psi` much more often than the symbol. So if looking for the “trident” symbol, search both `ψ` and `psi`.
  - archive-side wave/DASHI context is broader than the repo docs: cross-thread hits concentrate in `Branch · Topology and MDA/MDL`, `Interference and Learning Demo`, `Architectural Closure Status`, `Filament Fining Implementation`, and `Math Prof Outreach Stage`, where wave-lift / even-subalgebra / graded-series language exists mainly as archive context rather than local markdown.
  - sibling repo `../DASHIg` materially sharpens this picture:
    - `FORMALISM_OUTLINE.md` gives a researcher-facing formalism structure: ultrametric state space -> operator stack `T = P ∘ C ∘ R` -> contractive dynamics -> quadratic geometry -> cone/causality -> closure theorem -> empirical geometry.
    - `all_code44.txt` confirms local scaffolds for `DASHI.Physics.WaveLiftEvenSubalgebra`, `DASHI.Physics.OrbitShellGeneratingSeriesRootSystemB4`, `DASHI.Physics.Moonshine.FiniteGradedShellSeriesRootSystemB4`, `DASHI.Physics.Moonshine.FiniteTwinedShellTraceRootSystemB4`, `DASHI.Physics.LorentzNeighborhoodDynamicCandidate`, and explicit gauge-bridge / gauge-persistence exports.
    - the right reading is now: these objects are not merely archive rumors; they have at least sibling-repo Agda/module scaffolds, but they still do not amount to a finished local physics closure theorem in this repo.
  - ownership split sharpened by `../DASHIg` docs: this repo remains the concrete Python/benchmark/HEPData side, while the broader formalism write-up and many Agda-level closure/wave-lift structures live in `../DASHIg`.
- Resolved thread registry:
  - `689a818c-99d4-8330-aa60-5a3c484ef55a` -> `e40d17da77188861d419379aca12d1a3cdadc337` | `Compare 3-6-9 with TL` | source=`web` | topic: compare the 3-6-9 formalism with temporal logic and recursive time structure.
  - `690a9ea9-ce14-8324-bc56-4d4a9209732c` -> `a890929ced6b41f559536f34402245e98bbf92a1` | `Debates on causality` | source=`web` | topic: causality debate and tool-assisted argument cleanup.
  - `690acef3-7540-8320-9ad8-6e6d79b5231e` -> `863a5b9fad411141443ebf43f8f08d2e65c2a332` | `Cannabis reforms Australia 2026` | source=`web` | topic: Australia 2026 cannabis-reform lookup; out of repo scope.
  - `690e6469-9508-8320-86b4-669fe11d6245` -> `17bde1e6b2b7d785009b992bfaa1c4d74298dcb8` | `Quarter turn in quantum` | source=`web` | topic: quarter-turn operator / quantum rotation intuition.
  - `69166b40-4468-8320-8cc5-c7e7c45c576a` -> `40663b18f8aa5979c71cef8db3eb6b437ecaa510` | `Zip extraction hardening` | source=`web` | topic: path-traversal risk in `extractall`; out of current repo scope.
  - `6918ec5c-12b4-8324-8822-428dd6ddb04e` -> `df5aeb8fb883bf08d093c98649e6badbc90de4cb` | `Balanced ternary systems` | source=`web` | topic: balanced ternary representations and digit-set comparisons.
  - `691c71e3-9aec-8320-bd4f-56ba559ee7b8` -> `2933aaa349dc7453b21c2fce4a96bbb66b375748` | `Video summary attempt` | source=`web` | topic: summarize a linked video/paper and extract findings.
  - `691674c6-1154-8320-bf5a-facef5aa5f81` -> `6516d5174954dc5b11b1d8cc9e8b0d3b7d777b39` | `Watertight mesh repair` | source=`web` | topic: watertight mesh repair and undersuit-generation prerequisites.
  - `6909470b-9250-8324-961f-59559af5c6bd` -> `11a134a7c680f9cd5e4fe9d1be468f8cd21c23fd` | `seameinit` | source=`web` | topic: parametric bodysuit / Iron Man-style generator survey.
  - `6947ad7f-85f0-8322-8f18-4b34ed3f30ab` -> `508e20de4fa97307e2de88be9a49cd9b35838779` | `Origin of bra-ket notation` | source=`web` | topic: bra-ket notation origin; out of repo scope.
  - `695c4632-cc44-8320-96d5-e0309cf28a8a` -> `03f8adc17793b0ed854f7ec555f8e638613fa4cd` | `Wave-field Learning with Kernels` | source=`web` | topic: code direction for sparse wave-field learning with kernel methods.
  - `695f385c-fe88-8323-b414-c6479c6d2460` -> `de3adc952ae45282baee4c77887fae20e771b444` | `AI Expert Timeline Delay` | source=`web` | topic: AI-risk timeline article interpreted through the current formalism.
  - `6919bf75-af7c-8324-b2be-bfc2306d8208` -> `c5adc26f07706a65a5da6043eb91810f3041c9c0` | `P-adic quantum systems` | source=`web` | topic: p-adic trit/qubit algorithms and quantum encoding questions.
  - `696dbd1b-63a4-8320-9e24-440328c4fbf7` -> `88923ac659cb8f659d4477d8193e4213e11be121` | `Branch · Formalism Bridging GR and MDL` | source=`db` | topic: formalize the branch as definitions and theorems.
  - `696dbd23-8608-8320-90b7-57235c205bc0` -> `88923ac659cb8f659d4477d8193e4213e11be121` | `Branch · Formalism Bridging GR and MDL` | source=`db` | topic: write stochastic valuation-field equations explicitly.
  - `696dbd2b-5188-8324-92a7-e15f3033ffe1` -> `88923ac659cb8f659d4477d8193e4213e11be121` | `Branch · Formalism Bridging GR and MDL` | source=`db` | topic: map valuation curvature to Einstein equations explicitly.
  - `696dc067-d6e4-8324-b452-613333ba7501` -> `88923ac659cb8f659d4477d8193e4213e11be121` | `Branch · Formalism Bridging GR and MDL` | source=`db` | topic: identify KS density as an SU(3) class function.
  - `696dc06c-ec8c-8322-8247-87aeda019598` -> `88923ac659cb8f659d4477d8193e4213e11be121` | `Branch · Formalism Bridging GR and MDL` | source=`db` | topic: translate covariance, diffeomorphism, and tensor references into the house formalism.
  - `696dc063-ad00-8321-9e23-8efbb3858c88` -> `88923ac659cb8f659d4477d8193e4213e11be121` | `Branch · Formalism Bridging GR and MDL` | source=`db` | topic: exact M6 bitensor representation theory, weights, shells, and saturation.
  - `696ece79-7478-8324-b12a-e77ee0570eec` -> `8e946843bf256ce8961b128dc8b0c1968fa687e5` | `Branch · Branch · Formalism Bridging GR and MDL` | source=`db` | topic: add `(M_S, X_t)` and rerun MSSM Level 3.
  - `696ed143-2ddc-8324-9b6e-aced5c5954d8` -> `7024855e0c296911202e6318f2668aea816a7cb4` | `Branch · Branch · Branch · Formalism Bridging GR and MDL` | source=`db` | topic: deeper branch carrying the same GR/MDL translation set.
  - `69718c29-6bcc-8324-b9e9-e412af8c89eb` -> `53a59124cb8ef2f2e3a708a31fceb0010f3208ca` | `Branch · Topology and MDA/MDL` | source=`db` | topic: LES-related topology/MDA-MDL context, including tsunami-adjacent efficient-computation prompts.
  - `6978944d-f0f8-8321-b9ca-a4e6aee51db9` -> `ccef25ab501f5c95127ec0652bcf8c2119569c72` | `Filament Fining Implementation` | source=`db` | topic: implement filament/fining and preserve metrics/readback behavior.
  - `697c3293-9ed4-839a-ad1c-3367cdc388db` -> `5dd2ab979645898b8123ec5987b00d1489bf2e79` | `Spectral Gradient Implementation` | source=`db` | topic: match GPU LES behavior to the CPU spectral reference instead of generic GPU parity.
  - `696d865d-79a8-8321-a88c-2e37b963a4be` -> `d19835f24fda067c70c2f09935917a24b8a2ef42` | `Formalism Bridging GR and MDL` | source=`db` | topic: top-level bridge from GR language to the house valuation/MDL formalism.
  - `6990578b-a538-8399-aeab-e2979e6d4baa` -> `8730ad8f74a3d12cde0ce87829d503ad6f898abb` | `Math Mysticism Breakdown` | source=`web` | topic: separate numerology/mysticism claims from defensible mathematical structure.
  - `690e6e7a-ccd0-8321-b456-d27dba931120` -> `52cf44ec53a0624ef3ba631e4cbc4d0933dc4028` | `Quantum field count` | source=`web` | topic: question about field counts in quantum physics.
  - `696dbd54-8818-8324-b66b-70e7bdf32d2b` -> `47bd6cb79937a15ab269c3210728eb6b2308b078` | `Branch · Formalism Bridging GR and MDL - LES` | source=`db` | topic: LES-facing branch that ties the GR/MDL formalism to simulation work.
  - `6992b100-c460-839e-89b6-29abbddac25b` -> `54e662a4243d10d575758d394f3c472210ed7cd2` | `Branch · Math Mysticism Breakdown` | source=`web` | topic: branch cleanup continuing the math-vs-mysticism separation.
  - `69945d03-18d8-839b-b7bd-445af15e4812` -> `bd5cb7767e6c3a17517c9cde935e34edc2626442` | `Trit to bit encoding` | source=`web` | topic: ternary/trit-to-bit encoding.
  - `699454b7-d608-839a-ad7b-f8cd61ef6ed6` -> `2f7a03d39dca7e702ba7257fbc15143249705b36` | `File Analysis and Integration` | source=`web` | topic: integrate user-provided files and derived analysis into the working model.
  - `6993e106-93b4-83a1-930c-46ce77c1affe` -> `23fc78d959345d64cc8b6368bf3ce72a485375a0` | `Interference and Learning Demo` | source=`web` | topic: turn a local interference script into a learning/analysis demo.
  - `69984670-ff48-839a-8dee-65006e9986c9` -> `c3f6a80502ffd10c4a8817815fa854471dc3a9e7` | `Architectural Closure Status` | source=`web` | topic: ask whether the overall architecture has reached full closure.
  - `699dc65b-6510-839a-8b31-2ea717285a10` -> `cf05b8ab4bebd2247784e55f4d4be5df321c60fa` | `Cone monotonicity analysis` | source=`web` | topic: evaluate cone-premise violation rates across margins/eps settings.
  - `699dd0b2-1634-83a0-8bcc-4f24895331a4` -> `4746e7037080b37c87a1be47594a25a1b74351db` | `Branch · Branch · Cone monotonicity analysis` | source=`web` | topic: deeper cone-analysis branch extending the violation-rate exploration.
  - `699dc8f6-b6f0-839e-8b3a-7912abb07093` -> `64ca6555941802f7cd4974541eab012188b635b3` | `Branch · Cone monotonicity analysis` | source=`web` | topic: intermediate cone-analysis branch on the same premise set.
  - `699dd1e1-c760-839b-84cc-571972319794` -> `0380973ed34f2167750b0f48a1d3174197577270` | `Branch · Branch · Branch · Cone monotonicity analysis` | source=`web` | topic: deepest cone-analysis branch carrying the extended closure discussion.
  - `69aa52b4-6f7c-839f-aa7f-d120ffe0c1ad` -> `decf9e3cde5ccdec0c51ad8aab15999201503998` | `Math Prof Outreach Stage` | source=`db` | topic: closure-core mostly there but physics program still missing a natural dynamics law, conserved quantity, continuum/gauge recovery, and a documented wave-lift bridge.
