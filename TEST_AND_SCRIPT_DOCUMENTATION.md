# Test and Runner Script Inventory

This document summarizes the standalone visualization scripts (`pytest*.py`), the automated tests in `tests/`, and the CHSH/spectral-line tooling under `newtest/`.

## Gap: `Main_with_rotation` versus the documented helpers
The automation and exploratory scripts below all rely on small numerical utilities but none are wired into the top-level demo in
`dashifine/Main_with_rotation.py`. That module only emits placeholder slice/coarse PNGs and exposes minimal math helpers
(`gelu`, `orthonormalize`, `rotate_plane_4d`, palette blending, p-adic colouring). It is the target of the primitive/integration
tests and acts as a stub for the formal specs, but the visualization/CHSH runners described here do not exercise it directly.
Documenting this separation clarifies that feature work would need to route future rendering or CLI entry points through
`Main_with_rotation` to align the library with the example scripts and runners.

## Root `pytest*.py` visualization and analysis scripts
- **`pytest1.py`** builds two standing waves on a 2D grid, plots their superposition and the corresponding interference energy surface in 3D. 【F:pytest1.py†L1-L38】
- **`pytest2.py`** repeats the interference plots and adds a coupled-oscillator learning demo that fits sinusoid amplitudes/phases to a target waveform via stochastic gradient descent with Kuramoto-style coupling, logging trajectories and plotting convergence. 【F:pytest2.py†L1-L189】
- **`pytest3.py`** generates constructive/destructive modulus sequences, runs Hann-windowed FFTs, picks spectral peaks, and can save CSV plus overlay plots; helper utilities build the sequences, perform FFTs, pick peaks, and render outputs. 【F:pytest3.py†L1-L149】
- **`pytest4.py`** provides similar constructive/destructive FFT analysis with richer reporting: reference rational lines, peak-to-reference matching/enrichment, CSV export, and CLI parameters for spectrum sizing. 【F:pytest4.py†L1-L200】
- **`pytest5.py`** computes alignment strength (# of moduli dividing each integer), saves results to CSV, optionally plots smoothed/decimated curves, and prints top alignment points. 【F:pytest5.py†L1-L58】
- **`pytest6.py`** overlays alignment-strength curves for multiple modulus ranges, with normalization, smoothing/decimation, optional LCM markers, and plot saving. 【F:pytest6.py†L1-L93】

## Grokking scan scripts
- **`26_grok_critical_scan.py`** runs a critical-window grokking scan on the mod-multiplication task (`p=97` in the current configuration), logging per-run summaries and per-epoch trajectories. The current workflow:
  - writes `grok_critical_scan.csv` with `t_fit`, `t95`, final train/test loss, and final train/test accuracy
  - writes `grok_critical_scan_trajectories.csv` with per-epoch `train_loss`, `test_loss`, `train_acc`, and `test_acc`
  - resumes safely by skipping completed `(p, weight_decay, seed)` tuples already in the summary CSV
  - uses conservative early stopping only after 5 logged checkpoints in a row satisfy `test_acc >= 0.95`
  - is currently configured for a coarse onset scan over `weight_decay ∈ {0.25, 0.30, 0.35, 0.40}` with `seed=0` and no cross-prime runs
- **`26_grok_trajectory_analysis.py`** consumes the checkpointed grokking CSVs and produces first-pass analysis artifacts for curve-family inspection:
  - accepts one or more summary CSVs plus one or more trajectory CSVs so coarse and refinement scans can be analyzed together
  - milestone table (`t10`, `t20`, `t50`, `t80`, `t90`, `t95`) by `(p, weight_decay, seed)`
  - simple onset-fit screening table for `t95(weight_decay)`
  - raw trajectory overlays for `test_acc` and `test_loss`
  - normalized `test_acc` overlays using `epoch / t50` and `epoch / t95`
  - Gompertz shape-screening on normalized trajectories, including a shared-parameter fit summary and overlay plot
  - current result: on the 7-point dataset, the simple logistic baseline outperforms the shared-parameter Gompertz candidate on normalized test accuracy, so the shape law remains open
  - rising-phase-only logistic screening with a per-run onset shift, intended to test the “metastable plateau + shared post-escape rise” hypothesis more directly
  - current result: using a shared logistic rise on the post-`t10` segment with per-run onset shift gives `mse≈0.00119`, which is consistent with a shared sigmoid-like post-escape rise after the delayed plateau
  - rising-phase-only logistic screening on normalized post-`t10` test-loss progress
  - current loss-side result: normalized test-loss progress fits worse than accuracy (`mse≈0.00197`), so `test_acc` remains the cleaner empirical screen for the shared post-escape rise
  - fitted-onset logistic screening, which learns a separate onset shift for each run while sharing the post-escape logistic shape
  - current fitted-onset result: the shared logistic rise with learned onset shifts gives `mse≈0.000351`, materially better than the fixed-`t10` rise fit and essentially matching the best full normalized logistic baseline
  - lower-complexity onset comparisons:
    - fixed `t20` shift improves over fixed `t10` (`mse≈0.000714`) but does not match the fitted-onset screen
    - curvature-based onset performs poorly on the current dataset (`mse≈0.01474`)
  - shared normalized-onset screening:
    - a single shared onset `t0 = c * t50` with learned `c≈0.8055` gives `mse≈0.000360`
    - this is nearly identical to the per-run fitted-onset result (`mse≈0.000351`), so the onset alignment can be simplified without materially weakening the fit
- **`26_grok_critical_scan_refine.py`** is the lower-`weight_decay` follow-up runner. It mirrors the checkpoint/early-stop behavior of the main critical scan but uses a longer epoch budget and writes separate refinement outputs so the lower band can be tested without overwriting the coarse scan.
- **`26_grok_sweep.py`**, **`26_grok_sweep_2.py`**, **`26_grok_sweep_clean.py`**, and **`26_grok_sweep_adaptive.py`** remain useful for broader parameter sweeps, but they primarily save onset summaries rather than the full trajectories now required for curve-shape analysis.

## Automated tests in `tests/`
- **`tests/test_primitives.py`** validates numerical helpers in `dashifine.Main_with_rotation`: GELU oddness, orthonormalization, 4D/3D rotations, p-adic palette mapping, and class-weight to RGBA blending. 【F:tests/test_primitives.py†L1-L82】
- **`tests/test_integration.py`** runs `main` with tiny grids to ensure artifact paths are created. 【F:tests/test_integration.py†L1-L19】
- **`tests/test_chsh_harness.py`** checks that `two_qubit_from_two_local_planes` matches the expected Bell-state construction when rotating one local plane. 【F:tests/test_chsh_harness.py†L1-L25】
- **`tests/test_lattice_chsh.py`** perturbs lattice CHSH frames and asserts the CHSH score changes accordingly. 【F:tests/test_lattice_chsh.py†L1-L27】
- **`tests/test_lineage_palette.py`** exercises lineage palette helpers for address parsing and slice rendering, comparing against HSV mappings and explicit centre addresses. 【F:tests/test_lineage_palette.py†L1-L74】
- **`tests/test_runner_element_lines.py`** executes spectral-line CLI runners for dipole lines and FFT summaries, asserting CSV creation and expected headers. 【F:tests/test_runner_element_lines.py†L1-L56】

## `newtest/` quantum and spectral utilities
- **`chsh_harness.py`** defines Pauli-based CHSH utilities (measurement operators, Bell states, Tsirelson angles, CHSH score) plus helpers to extract local plane bases, rotate vectors, build projectors, and map two local planes into a maximally entangled two-qubit state. 【F:newtest/chsh_harness.py†L1-L169】
- **`chsh_extras.py`** collects CHSH-related calculations: Pauli matrices, measurement operators, Kronecker products, density-matrix helpers, depolarization, expectation evaluation, projector construction, Horodecki S_max scans, frame unitaries, and fixed-frame CHSH scoring utilities. 【F:newtest/chsh_extras.py†L1-L242】
- **`lattice_chsh.py`** implements lattice-oriented CHSH tooling: local basis extraction, Pauli/measurement primitives, Bell states, CHSH scorer, Tsirelson angles, Heisenberg unitaries, lattice leg builders (open/quantized/gaussian/composite variants), and full lattice CHSH evaluation. 【F:newtest/lattice_chsh.py†L1-L328】
- **`triality_stack.py`** contains triality-stack Hamiltonian and phase utilities: rotation helper `R`, eigenvalue sorting, open-leg Hamiltonian builders, chiral/γ-operator constructors, phase quantization/randomization/gaussian noise, composite-moduli phase generation, and LCM utilities. 【F:newtest/triality_stack.py†L1-L394】
- **`runner_triality_chsh.py`** scans triality-phase triplets for CHSH values, optional plotting, and combines helper routines for phase generation and CHSH evaluation. 【F:newtest/runner_triality_chsh.py†L1-L211】
- **`runner_phase_locked_entanglers.py`** prepares Bell-like states, computes CHSH S for leg pairs, scans τ, builds heatmaps, and offers a CLI to run/plot the sweep. 【F:newtest/runner_phase_locked_entanglers.py†L1-L217】
- **`runner_phase_offset_scan.py`** defines small rotation helpers, random leg-frame builders, state preparation, CHSH-with-offset evaluation, τ/δ scans with FFT analysis, plotting helpers, and a CLI entry point. 【F:newtest/runner_phase_offset_scan.py†L1-L208】
- **`runner_synced_entangler.py`** offers synced entangler construction, random quantized leg generation, CHSH computation for frames, and a main entry point. 【F:newtest/runner_synced_entangler.py†L1-L152】
- **`runner_decoherence_heatmap.py`**, **`runner_overlay_decoherence.py`**, **`runner_discrete_decoherence.py`**, **`runner_discrete_decoherence_avg.py`**, and **`runner_discrete_decoherence_frames.py`** provide various decoherence scans (continuous, overlayed, discrete, averaged, or frame-by-frame), all exposing CLIs built around CHSH score evaluations. 【F:newtest/runner_decoherence_heatmap.py†L1-L116】【F:newtest/runner_overlay_decoherence.py†L1-L140】【F:newtest/runner_discrete_decoherence.py†L1-L43】【F:newtest/runner_discrete_decoherence_avg.py†L1-L100】【F:newtest/runner_discrete_decoherence_frames.py†L1-L58】
- **`runner_lattice_chsh.py`**, **`runner_mixed_modulus.py`**, **`runner_composite_moduli.py`**, and **`runner_cross_moduli_compare.py`** wrap the lattice/phase-modulus experiments in CLI scripts for sampling CHSH behavior under different modulus/phase configurations. 【F:newtest/runner_lattice_chsh.py†L1-L49】【F:newtest/runner_mixed_modulus.py†L1-L58】【F:newtest/runner_composite_moduli.py†L1-L162】【F:newtest/runner_cross_moduli_compare.py†L1-L149】
- **`runner_chsh_experiments.py`** is a lightweight CLI for running canned CHSH experiments defined in the module. 【F:newtest/runner_chsh_experiments.py†L1-L46】
- **`runner_ternary_chsh.py`** drives ternary CHSH demonstrations from the command line. 【F:newtest/runner_ternary_chsh.py†L1-L20】
- **`runner_quantum_defect_demo.py`**, **`quantum_defect.py`**, and **`map_27_to_H3x3.py`** cover quantum-defect demo plumbing and small helpers for mapping 27-component weights into 3×3 coarse representations. 【F:newtest/runner_quantum_defect_demo.py†L1-L18】【F:newtest/quantum_defect.py†L1-L6】【F:newtest/map_27_to_H3x3.py†L1-L28】
- **`ternary_hilbert.py`** defines ternary/qutrit basis matrices and embedding utilities (`omega`, `basis_H3`, `Z_qutrit`, `X_qutrit`, `phase_form_matrix`, `embed_qubit_plane`, `kron3`, `modM_phase_rotation`). 【F:newtest/ternary_hilbert.py†L1-L66】
- **`motifs9.py`** provides triplet-to-motif binning helpers, coarse-graining from 27→9 bins, heatmap building, and a CLI-oriented `main`. 【F:newtest/motifs9.py†L1-L62】
- **`embed_chsh_ternary.py`** embeds qubit CHSH operators/states into a qutrit space with utilities for rotations, CHSH operator construction, Bell-state definition, and expectation evaluation. 【F:newtest/embed_chsh_ternary.py†L1-L80】
- **`analyze_tau_delta_coupling.py`** loads coupling scan arrays, smooths data, computes FFTs over τ/δ axes, builds slope maps, and exposes a CLI `main`. 【F:newtest/analyze_tau_delta_coupling.py†L1-L239】
- **Hydrogenic spectrum helpers**: `hydrogenic_numerov.py` computes energy levels/gaps and derived wavelengths/frequencies; `hydrogenic.py` contains finite-difference kinetic/potential builders; `quantum_defect.py` holds minimal stubs. 【F:newtest/hydrogenic_numerov.py†L1-L150】【F:newtest/hydrogenic.py†L1-L48】【F:newtest/quantum_defect.py†L1-L6】
- **Spectral line generation and runners**: `element_lines.py` defines hydrogenic states, oscillator-strength proxy, line record construction, and `build_dipole_allowed_lines`; `runner_element_lines.py` parses atomic-number arguments, generates line CSVs, and writes safely; `runner_lines_fft.py` converts line intensities to FFT summaries with CSV output. 【F:newtest/element_lines.py†L1-L122】【F:newtest/runner_element_lines.py†L1-L125】【F:newtest/runner_lines_fft.py†L1-L147】
- **Hydrogenic demos**: `runner_hydrogenic_demo.py` and `runner_hydrogenic_numerov.py` expose minimal CLIs for sample spectra/levels. 【F:newtest/runner_hydrogenic_demo.py†L1-L12】【F:newtest/runner_hydrogenic_numerov.py†L1-L14】
- **Legacy line pairing**: `runner_element_lines_0.py` enumerates transitions from levels; `runner_element_lines_1.py` pairs lines for combined analysis. 【F:newtest/runner_element_lines_0.py†L1-L47】【F:newtest/runner_element_lines_1.py†L1-L51】
- **Composite/modulus utilities**: `motifs9.py`, `runner_mixed_modulus.py`, and `runner_composite_moduli.py` explore composite-modulus spectra and motif aggregation; `map_27_to_H3x3.py` supports coarse mapping. 【F:newtest/motifs9.py†L1-L62】【F:newtest/runner_mixed_modulus.py†L1-L58】【F:newtest/runner_composite_moduli.py†L1-L162】【F:newtest/map_27_to_H3x3.py†L1-L28】
- **Miscellaneous**: `runner_cross_moduli_compare.py` contrasts cross-moduli outputs; `runner_overlay_decoherence.py` overlays decoherence channels; `runner_decoherence_heatmap.py` builds CHSH heatmaps; `runner_lattice_chsh.py` runs lattice CHSH demos; `runner_chsh_experiments.py` and `runner_ternary_chsh.py` provide additional experiment drivers. 【F:newtest/runner_cross_moduli_compare.py†L1-L149】【F:newtest/runner_overlay_decoherence.py†L1-L140】【F:newtest/runner_decoherence_heatmap.py†L1-L116】【F:newtest/runner_lattice_chsh.py†L1-L49】【F:newtest/runner_chsh_experiments.py†L1-L46】【F:newtest/runner_ternary_chsh.py†L1-L20】
