#!/usr/bin/env python3
"""
26_hepdata_proof_dossier.py

Creates a "proof dossier" (empirical certificates) for:
A) Contraction: estimate Lipschitz constants of your contraction flow(s)
B) Quadratic form: find best-fit symmetric bilinear form G such that J^T G J ≈ λ G locally
C) Signature: estimate sign pattern / effective rank of G (bootstrap stability)

Inputs expected:
  - hepdata_to_dashi/<obs>/lenses_continuous.csv   (bin, L0..L9)
  - hepdata_dashi_native/<obs>_dashi_native_metrics.csv  (iter, alpha, ..., b0..b4)

Outputs:
  - proof_dossier/report.json
  - proof_dossier/report.md
  - proof_dossier/plots/*.png
"""

from __future__ import annotations
import os, json, math, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
LENS_ROOT_DEFAULT = "hepdata_to_dashi"
BETA_ROOT_DEFAULT = "hepdata_dashi_native"
OUT_DEFAULT = "proof_dossier"

# contraction checks
PAIR_SAMPLES = 5000      # random pairs for Lipschitz estimation
RNG_SEED = 0

# local Jacobian estimation for quadratic-form fit
J_SAMPLES = 200          # number of local Jacobians to estimate
J_EPS = 1e-3             # finite-difference step in lens space
NEIGH_K = 8              # neighbors used for local linear map fit

# bootstrap for signature stability
BOOT_REPS = 50
BOOT_FRAC = 0.75

# numerical
RIDGE = 1e-8

# -------------------------
# Helpers
# -------------------------
def read_lens_space(lens_root: str) -> tuple[np.ndarray, list[str], list[tuple[str,int]]]:
    """
    Stacks all continuous lens vectors across observables.
    Returns:
      X: (N,10)
      obs_names
      index_map: list of (obs, bin_index) per row
    """
    lens_root = str(lens_root)
    obs_dirs = sorted([p for p in Path(lens_root).iterdir() if p.is_dir()])
    all_rows = []
    index_map = []
    obs_names = []
    for od in obs_dirs:
        f = od / "lenses_continuous.csv"
        if not f.exists():
            continue
        data = np.genfromtxt(f, delimiter=",", names=True)
        # columns: bin, L0..L9
        cols = [f"L{k}" for k in range(10)]
        X = np.column_stack([data[c] for c in cols]).astype(float)
        # drop NaNs
        mask = np.all(np.isfinite(X), axis=1)
        X = X[mask]
        all_rows.append(X)
        obs_names.append(od.name)
        # record bin indices (after masking, approximate)
        bins = np.arange(len(mask))[mask]
        for b in bins:
            index_map.append((od.name, int(b)))
    if not all_rows:
        raise SystemExit(f"No lenses_continuous.csv found under {lens_root}")
    X = np.vstack(all_rows)
    return X, obs_names, index_map

def whiten(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    C = np.cov(X.T)
    C = 0.5*(C + C.T)
    # ridge
    C = C + RIDGE*np.trace(C)/max(1, C.shape[0])*np.eye(C.shape[0])
    L = np.linalg.cholesky(C)
    Xw = np.linalg.solve(L, (X - mu).T).T
    return Xw, mu, L

def pairwise_lipschitz(X0: np.ndarray, X1: np.ndarray, metric="l2", rng=None) -> dict:
    """
    Estimate Lipschitz constant of map X0 -> X1 on a shared point set by random pairs:
      L ≈ max ||f(x)-f(y)|| / ||x-y||
    metric:
      - "l2": Euclidean
      - "linf": max norm
      - "ultra": ultrametric proxy via agreement depth on ternary-coded signs
    """
    assert X0.shape == X1.shape
    n = X0.shape[0]
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    idx_a = rng.integers(0, n, size=PAIR_SAMPLES)
    idx_b = rng.integers(0, n, size=PAIR_SAMPLES)
    # avoid identical pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n

    A0 = X0[idx_a]; B0 = X0[idx_b]
    A1 = X1[idx_a]; B1 = X1[idx_b]

    if metric == "l2":
        d0 = np.linalg.norm(A0 - B0, axis=1)
        d1 = np.linalg.norm(A1 - B1, axis=1)
    elif metric == "linf":
        d0 = np.max(np.abs(A0 - B0), axis=1)
        d1 = np.max(np.abs(A1 - B1), axis=1)
    elif metric == "ultra":
        # ultrametric proxy: encode sign pattern of coordinates (ternary-like: -1/0/+1 by tau)
        tau = 0.25
        def tri(Z):
            T = np.zeros_like(Z, dtype=np.int8)
            T[Z > tau] = 1
            T[Z < -tau] = -1
            return T
        T0a = tri(A0); T0b = tri(B0)
        T1a = tri(A1); T1b = tri(B1)
        # "agreement depth" proxy = number of matching coords
        agree0 = (T0a == T0b).sum(axis=1)
        agree1 = (T1a == T1b).sum(axis=1)
        # convert to distance: larger agree => smaller distance, use 2^{-agree}
        d0 = 2.0 ** (-agree0.astype(float))
        d1 = 2.0 ** (-agree1.astype(float))
    else:
        raise ValueError("unknown metric")

    eps = 1e-12
    ratios = d1 / (d0 + eps)
    ratios = ratios[np.isfinite(ratios)]
    return dict(
        metric=metric,
        max=float(np.max(ratios)),
        p99=float(np.quantile(ratios, 0.99)),
        p95=float(np.quantile(ratios, 0.95)),
        median=float(np.median(ratios)),
        n=int(len(ratios)),
    )

# -------------------------
# Local Jacobians (data-driven) and quadratic form fit
# -------------------------
def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """
    Simple exact kNN via squared distances (N is small ~81).
    Returns idx (N,k) excluding self.
    """
    N = X.shape[0]
    D = np.sum((X[:,None,:] - X[None,:,:])**2, axis=2)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]

def local_linear_map(X: np.ndarray, F: np.ndarray, i: int, nbrs: np.ndarray) -> np.ndarray:
    """
    Fit local linear map J at point i: (F - Fi) ≈ J (X - Xi) using neighbors.
    Returns J (dxd).
    """
    Xi = X[i]; Fi = F[i]
    Xn = X[nbrs] - Xi
    Fn = F[nbrs] - Fi
    # solve Fn ≈ Xn @ J^T  => J^T = lstsq(Xn, Fn)
    JT, *_ = np.linalg.lstsq(Xn, Fn, rcond=None)
    return JT.T

def fit_quadratic_form(Js: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """
    Find symmetric G (dxd) minimizing sum ||J^T G J - λ G||_F^2
    We solve a linear least squares in vec(G) with a shared λ (one scalar).
    Approach:
      Let g be vec(G) (d^2). For each J, constraint:
        vec(J^T G J) - λ vec(G) ≈ 0
      => (K(J) - λ I) g ≈ 0 where K(J)= (J^T ⊗ J^T?) Actually:
         vec(J^T G J) = (J^T ⊗ J^T?) vec(G) = (J ⊗ J)^T vec(G)
      We'll compute Mj = kron(J, J) and use (Mj - λ I) g ≈ 0.
    Then:
      minimize over (g,λ) with ||g||=1. We do a 1D search over λ, choose λ giving smallest
      smallest singular value of stacked matrix.
    Enforce symmetry of G after reshape.
    """
    d = Js[0].shape[0]
    I = np.eye(d*d)
    # precompute kron(J, J)
    Ms = [np.kron(J, J) for J in Js]
    # search λ on a reasonable range (contractions => |λ|<1)
    lam_grid = np.linspace(-0.5, 0.99, 300)
    best = None
    for lam in lam_grid:
        A = np.vstack([M - lam*I for M in Ms])  # (m*d^2, d^2)
        # smallest right singular vector
        try:
            _, s, Vt = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        g = Vt[-1]
        resid = float(s[-1])
        if best is None or resid < best[0]:
            best = (resid, float(lam), g)
    if best is None:
        raise RuntimeError("failed to fit quadratic form")
    resid, lam, g = best
    G = g.reshape(d, d)
    # symmetrize
    G = 0.5*(G + G.T)
    # normalize
    G = G / (np.linalg.norm(G) + 1e-12)
    return G, lam

def signature(G: np.ndarray, tol: float = 1e-6) -> dict:
    w = np.linalg.eigvalsh(0.5*(G+G.T))
    pos = int((w > tol).sum())
    neg = int((w < -tol).sum())
    zer = int(len(w) - pos - neg)
    return dict(pos=pos, neg=neg, zero=zer, evals=w.tolist())

# -------------------------
# Beta-flow ingestion (true DASHI-native contraction)
# -------------------------
def read_beta_flows(beta_root: str) -> dict[str, np.ndarray]:
    """
    Reads *_dashi_native_metrics.csv and returns dict:
      label -> betas (T,5)
    """
    out = {}
    for f in sorted(Path(beta_root).glob("*_dashi_native_metrics.csv")):
        data = np.genfromtxt(f, delimiter=",", names=True)
        # columns b0..b4
        betas = np.column_stack([data[f"b{k}"] for k in range(5)]).astype(float)
        out[f.name.replace("_dashi_native_metrics.csv","")] = betas
    return out

# -------------------------
# Main report
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-root", default=LENS_ROOT_DEFAULT)
    ap.add_argument("--beta-root", default=BETA_ROOT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.out)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    # ---------
    # Load lens space
    # ---------
    X, obs_names, index_map = read_lens_space(args.lens_root)
    N, d = X.shape
    Xw, mu, L = whiten(X)

    # ---------
    # Load beta flows (true contraction in coefficient space)
    # ---------
    flows = read_beta_flows(args.beta_root)

    report = {
        "lens_space": {"N": int(N), "d": int(d), "observables": obs_names},
        "contraction": {},
        "quadratic_form": {},
        "beta_flow": {},
    }

    # ---------
    # A) Contraction certificates for beta flows
    # ---------
    for label, B in flows.items():
        # map between successive steps: Bt -> Bt+1
        # Lipschitz estimate on the set of points themselves (pairwise ratios between consecutive time slices)
        # We'll treat X0=Bt and X1=Bt+1 for each t and report max/p99.
        lips = []
        for t in range(B.shape[0]-1):
            X0 = B
            # use the SAME point set trick: define map f_t that moves only point t? too few points.
            # Instead: measure contraction of trajectory increments:
            # ratios of step lengths: ||Δ_{t+1}|| / ||Δ_t|| as a 1D Lipschitz proxy.
            pass

    # Better: for beta-flow, the clean contraction check is on the *step map* applied to arbitrary beta.
    # But we only have observed betas. So we certify contraction of the trajectory itself:
    # - distance to final fixed point shrinks
    # - successive distances shrink
    beta_flow_report = {}
    for label, B in flows.items():
        T = B.shape[0]
        b_star = B[-1]
        dist = np.linalg.norm(B - b_star, axis=1)
        # ratios dist_{t+1}/dist_t
        ratios = []
        for t in range(T-1):
            if dist[t] > 1e-12:
                ratios.append(dist[t+1]/dist[t])
        beta_flow_report[label] = {
            "T": int(T),
            "dist_to_final": dist.tolist(),
            "ratio_median": float(np.median(ratios)) if ratios else float("nan"),
            "ratio_max": float(np.max(ratios)) if ratios else float("nan"),
        }
    report["beta_flow"] = beta_flow_report

    # plot one representative beta distance curve
    if beta_flow_report:
        lab0 = sorted(beta_flow_report.keys())[0]
        dist = np.array(beta_flow_report[lab0]["dist_to_final"])
        plt.figure(figsize=(9,4))
        plt.plot(np.arange(len(dist)), dist, marker="o")
        plt.grid(True)
        plt.xlabel("iteration")
        plt.ylabel("||beta_t - beta_*||")
        plt.title(f"Beta contraction evidence: {lab0}")
        plt.tight_layout()
        plt.savefig(outdir/"plots"/"beta_dist_to_fixedpoint.png", dpi=160)
        plt.close()

    # ---------
    # A) Contraction certificates in lens space (raw/white/ultra proxy)
    # We need an actual map f. We don’t have T on lens vectors yet, so we certify
    # "self-consistency contraction" by projecting to low-d manifold and using denoising map:
    # f(x) = kNN mean in whitened space (a proxy for projection operator P).
    # This is explicitly labelled as a *proxy* unless you plug your actual operator in.
    # ---------
    nbrs = knn_indices(Xw, NEIGH_K)
    Fw = np.zeros_like(Xw)
    for i in range(N):
        Fw[i] = Xw[nbrs[i]].mean(axis=0)
    # unwhiten back to raw coordinates: X = mu + L @ Xw
    F = (mu + (L @ Fw.T).T)

    # Lipschitz on proxy P
    report["contraction"]["proxy_projection"] = {
        "raw_l2": pairwise_lipschitz(X, F, metric="l2", rng=rng),
        "raw_linf": pairwise_lipschitz(X, F, metric="linf", rng=rng),
        "raw_ultra": pairwise_lipschitz(X, F, metric="ultra", rng=rng),
        "white_l2": pairwise_lipschitz(Xw, Fw, metric="l2", rng=rng),
    }

    # ---------
    # B) Quadratic form extraction on the proxy map in lens space
    # Estimate local Jacobians J_i from neighbors: F - Fi ≈ J (X - Xi)
    # Then fit symmetric G such that J^T G J ≈ λ G across samples.
    # ---------
    # sample indices
    idx = rng.choice(N, size=min(J_SAMPLES, N), replace=False)
    Js = []
    for i in idx:
        J = local_linear_map(Xw, Fw, i, nbrs[i])  # operate in whitened coords for stability
        Js.append(J)

    G, lam = fit_quadratic_form(Js)
    sig = signature(G)

    # bootstrap signature stability
    boot_sigs = []
    for _ in range(BOOT_REPS):
        m = max(5, int(len(Js)*BOOT_FRAC))
        sub = [Js[j] for j in rng.choice(len(Js), size=m, replace=True)]
        Gb, lamb = fit_quadratic_form(sub)
        boot_sigs.append(signature(Gb))
    # count most common (pos,neg,zero)
    trip = [(s["pos"], s["neg"], s["zero"]) for s in boot_sigs]
    uniq, cnt = np.unique(np.array(trip), axis=0, return_counts=True)
    order = np.argsort(-cnt)
    top = [(tuple(map(int, uniq[i])), int(cnt[i])) for i in order[:5]]

    report["quadratic_form"]["lens_space_proxy"] = {
        "space": "whitened",
        "lambda": float(lam),
        "G": G.tolist(),
        "signature": sig,
        "bootstrap_top_signatures": top,
        "note": "This G is fitted for the proxy projection map (kNN-mean). Swap Fw for your true DASHI map on lens vectors to make this a direct test.",
    }

    # plot eigenvalues of G
    w = np.array(sig["evals"])
    plt.figure(figsize=(9,4))
    plt.plot(np.arange(1,len(w)+1), np.sort(w)[::-1], marker="o")
    plt.grid(True)
    plt.xlabel("eigen index")
    plt.ylabel("eigenvalue")
    plt.title("Fitted invariant quadratic form G: eigen spectrum (whitened lens space)")
    plt.tight_layout()
    plt.savefig(outdir/"plots"/"G_eigs.png", dpi=160)
    plt.close()

    # write JSON + a readable markdown
    (outdir).mkdir(parents=True, exist_ok=True)
    with open(outdir/"report.json","w",encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # simple markdown summary
    md = []
    md.append("# DASHI Proof Dossier (Empirical Certificates)\n")
    md.append(f"- Lens points: **N={N}**, ambient **d={d}**\n")
    md.append(f"- Observables: {', '.join(obs_names)}\n")
    md.append("## A) Contraction\n")
    md.append("### Lens-space proxy projection (kNN-mean)\n")
    for k,v in report["contraction"]["proxy_projection"].items():
        md.append(f"- **{k}**: max={v['max']:.3f}  p99={v['p99']:.3f}  p95={v['p95']:.3f}  median={v['median']:.3f}\n")
    md.append("\n### Beta-flow contraction (true DASHI-native coefficient flow)\n")
    for lab, r in beta_flow_report.items():
        md.append(f"- **{lab}**: ratio_median={r['ratio_median']:.3g}  ratio_max={r['ratio_max']:.3g}\n")
    md.append("\n## B) Quadratic-form (data-driven)\n")
    q = report["quadratic_form"]["lens_space_proxy"]
    md.append(f"- Fitted λ ≈ **{q['lambda']:.4f}** (target is |λ|<1 under contraction)\n")
    sig = q["signature"]
    md.append(f"- Signature(G) ≈ **(+{sig['pos']}, −{sig['neg']}, 0:{sig['zero']})** (tol=1e-6)\n")
    md.append(f"- Bootstrap top signatures: {q['bootstrap_top_signatures']}\n")
    md.append("\n## Plots\n")
    md.append("- plots/G_eigs.png\n")
    md.append("- plots/beta_dist_to_fixedpoint.png\n")

    with open(outdir/"report.md","w",encoding="utf-8") as f:
        f.write("".join(md))

    print(f"\nWrote: {outdir/'report.json'}")
    print(f"Wrote: {outdir/'report.md'}")
    print(f"Wrote plots to: {outdir/'plots'}")
    print("\nNOTE:")
    print(" - Lens-space contraction + quadratic-form fitting is currently on a PROXY projection map (kNN mean).")
    print(" - To make it fully DASHI-native in lens space, replace Fw with the output of your true T-map on lens vectors.")
    print(" - Beta-flow contraction check IS on your true penalized-GLS RG flow outputs.")

if __name__ == "__main__":
    main()
