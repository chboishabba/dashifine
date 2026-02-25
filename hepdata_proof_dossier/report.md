# DASHI proof dossier (HEPData lens manifold)

Inputs:
- lens_root: `hepdata_to_dashi`
- beta_root: `hepdata_dashi_native`
- n=81, d=10
- observables: ['pTll_106_170', 'pTll_170_350', 'pTll_50_76', 'pTll_76_106', 'phistar_50_76']

## Headline metrics

| space | TwoNN | MLE | PCA cum3 | H1 max | H2 max |
|---|---:|---:|---:|---:|---:|
| RAW | 2.912 | 3.754 | 0.719 | 0.5245 | 0.1288 |
| WHITENED | 3.079 | 5.321 | 0.300 | 0.7291 | 0.1544 |
| SUB(k=4) | 3.599 | 2.768 | 0.928 | 0.3672 | 0.0776 |
| SUBW(k=4) | 4.674 | 2.609 | 0.750 | 0.5799 | 0.0444 |

## Null comparison (Gaussian, RAW)

- TwoNN(null) = 7.62 ± 1.07

- H1(null)    = 0.364 ± 0.0688

- Z: TwoNN(real vs null) = -4.4σ

- Z: H1(real vs null)    = 2.33σ

## Delta-cone signature screen (per-step embedding)

- Embedding columns: `v_pnorm`, `v_dnorm`, `v_depth`, `v_arrow`.
- Best indefinite mask (diagonal ±1 on `[v_pnorm, v_dnorm, v_arrow]`): `(+,-,-)` with `(p,q,z)=(1,2,0)`.
- Forward-step cone fraction: 59/60 (min per-label forward fraction ≈ 0.9167).
- Single violation: `pTll_76_106` at iter 9 → 10.
  - Δv_pnorm = -2.7747815774
  - Δv_dnorm ≈ -2.20e-11 (near-null axis in Δ-space)
  - Δv_arrow = -2.4908590599
  - Q(Δx) ≈ 1.495034 > 0 (spacelike under this signature)

Interpretation: one discrete jump dominates the miss; the remaining forward steps satisfy the Lorentz candidate. The Δv_dnorm axis behaves as effectively null in this step.


## Where each of the 10 dims comes from

In this run, each dimension is one **continuous lens channel** emitted by your `project_to_field_first` pipeline.
So `d=10` means you wrote a 10-vector per sample: `lens_0..lens_9`.
If you want semantic names (Self/Norm/Mirror × time), wire them into the NPZ key metadata and this script will surface them.


## Output files

- report.json
- summary.csv
- pca_evr_raw.png / pca_evr_whitened.png
- isomap_resid_raw.png / isomap_resid_whitened.png
- diffusion_eigs_raw.png / diffusion_eigs_whitened.png
- persistence_diagrams_raw.png / persistence_diagrams_whitened.png
