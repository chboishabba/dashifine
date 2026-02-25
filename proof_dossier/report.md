# DASHI Proof Dossier (Empirical Certificates)
- Lens points: **N=81**, ambient **d=10**
- Observables: pTll_106_170, pTll_170_350, pTll_50_76, pTll_76_106, phistar_50_76
## A) Contraction
### Lens-space proxy projection (kNN-mean)
- **raw_l2**: max=1.695  p99=1.109  p95=0.931  median=0.647
- **raw_linf**: max=1.330  p99=1.054  p95=0.905  median=0.639
- **raw_ultra**: max=64.000  p99=16.000  p95=8.000  median=1.000
- **white_l2**: max=1.029  p99=0.898  p95=0.808  median=0.560

### Beta-flow contraction (true DASHI-native coefficient flow)
- **pTll_106_170**: ratio_median=0.991  ratio_max=1
- **pTll_170_350**: ratio_median=0.973  ratio_max=1
- **pTll_50_76**: ratio_median=0.993  ratio_max=1
- **pTll_76_106**: ratio_median=1  ratio_max=1.04
- **phistar_50_76**: ratio_median=0.998  ratio_max=1.01

## B) Quadratic-form (data-driven)
- Fitted λ ≈ **0.0432** (target is |λ|<1 under contraction)
- Signature(G) ≈ **(+5, −5, 0:0)** (tol=1e-6)
- Bootstrap top signatures: [((5, 5, 0), 31), ((4, 6, 0), 10), ((6, 4, 0), 8), ((5, 4, 1), 1)]

## Plots
- plots/G_eigs.png
- plots/beta_dist_to_fixedpoint.png
