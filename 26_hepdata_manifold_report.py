#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# PH
from ripser import ripser

# sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors


# ---------------------------
# Basic utilities
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def stable_cov(X: np.ndarray, eps_scale: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mu, C, L) where C is stabilized covariance and L is Cholesky(C).
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    C = np.cov(X.T)
    C = 0.5 * (C + C.T)
    eps = eps_scale * (np.trace(C) / max(1, C.shape[0]))
    C = C + eps * np.eye(C.shape[0])
    L = np.linalg.cholesky(C)
    return mu, C, L

def whiten_with_map(X: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    Xc = X - mu
    return np.linalg.solve(L, Xc.T).T

def whiten_mahalanobis(X: np.ndarray, eps_scale: float = 1e-6) -> np.ndarray:
    mu, C, L = stable_cov(X, eps_scale=eps_scale)
    return whiten_with_map(X, mu, L)

def max_persistence(dgm: np.ndarray) -> float:
    if dgm is None or len(dgm) == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[np.isfinite(pers)]
    if len(pers) == 0:
        return 0.0
    return float(np.max(pers))

def compute_ph_max(X: np.ndarray, maxdim: int = 2) -> Tuple[float, float, float]:
    out = ripser(X, maxdim=maxdim)
    dgms = out["dgms"]
    H0 = max_persistence(dgms[0])
    H1 = max_persistence(dgms[1]) if len(dgms) > 1 else 0.0
    H2 = max_persistence(dgms[2]) if len(dgms) > 2 else 0.0
    return H0, H1, H2

def pca_spectrum(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA().fit(X)
    return pca.explained_variance_ratio_, pca.singular_values_


# ---------------------------
# Intrinsic dimension estimators
# ---------------------------

def twonn_dimension(X: np.ndarray) -> float:
    """
    Facco et al. TwoNN: m = 1 / mean(log(r2/r1))
    """
    X = np.asarray(X, float)
    if len(X) < 8:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mu = r2 / (r1 + 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 0)]
    if len(mu) == 0:
        return float("nan")
    return float(1.0 / (np.mean(np.log(mu)) + 1e-12))

def levina_bickel_mle_dimension(X: np.ndarray, k: int = 10) -> float:
    """
    Levina–Bickel MLE intrinsic dimension.
    For each point i:
      m_i = (k-1) / sum_{j=1..k-1} log(T_k / T_j)
    Return mean m_i.
    """
    X = np.asarray(X, float)
    n = len(X)
    if n < max(20, k + 2):
        return float("nan")
    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    # dists[:,0] = 0 self
    Tk = dists[:, k_eff]
    logs = []
    for j in range(1, k_eff):
        Tj = dists[:, j]
        ratio = (Tk + 1e-12) / (Tj + 1e-12)
        logs.append(np.log(ratio))
    denom = np.sum(np.stack(logs, axis=1), axis=1)
    denom = np.where(denom <= 1e-12, np.nan, denom)
    m_i = (k_eff - 1) / denom
    return float(np.nanmean(m_i))


# ---------------------------
# Diffusion maps (lightweight)
# ---------------------------

def diffusion_maps_eigs(X: np.ndarray, n_eigs: int = 10, sigma: Optional[float] = None) -> np.ndarray:
    """
    Simple diffusion maps: build RBF affinity, row-normalize to Markov, take eigenvalues.
    For stability at small n=81 this is fine.
    """
    X = np.asarray(X, float)
    n = len(X)
    if n < 5:
        return np.array([], float)

    # pairwise squared distances
    G = np.sum(X * X, axis=1, keepdims=True)
    D2 = G + G.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)

    # sigma heuristic: median distance
    if sigma is None:
        tri = D2[np.triu_indices(n, 1)]
        med = np.sqrt(np.median(tri) + 1e-12)
        sigma = float(med) if med > 0 else 1.0

    K = np.exp(-D2 / (2.0 * sigma * sigma + 1e-12))
    # Markov normalize
    row = K.sum(axis=1, keepdims=True)
    P = K / (row + 1e-12)

    # eigenvalues of P (largest first)
    # n small -> full eig ok
    evals = np.linalg.eigvals(P)
    evals = np.real(evals)
    evals = np.sort(evals)[::-1]
    return evals[: min(n_eigs, len(evals))]


# ---------------------------
# Isomap residual variance curve
# ---------------------------

def isomap_residual_curve(X: np.ndarray, dims: List[int], n_neighbors: int = 10) -> Dict[int, float]:
    """
    Fit Isomap for each target dim and compute 1 - R^2 between
    original geodesic distances (Isomap stores it) and embedded Euclidean distances.
    Lower is better (closer to an isometric embedding).
    """
    X = np.asarray(X, float)
    out = {}
    for d in dims:
        d_eff = min(d, X.shape[1], max(2, len(X) - 1))
        iso = Isomap(n_neighbors=min(n_neighbors, len(X) - 1), n_components=d_eff)
        Y = iso.fit_transform(X)
        # geodesic distance matrix from isomap
        G = iso.dist_matrix_
        # embedded euclidean distance matrix
        H = np.sum(Y * Y, axis=1, keepdims=True)
        E2 = H + H.T - 2.0 * (Y @ Y.T)
        E2 = np.maximum(E2, 0.0)
        E = np.sqrt(E2 + 1e-12)

        # correlate upper-triangles
        iu = np.triu_indices(len(X), 1)
        g = G[iu].ravel()
        e = E[iu].ravel()
        # R^2
        cg = g - g.mean()
        ce = e - e.mean()
        denom = (np.linalg.norm(cg) * np.linalg.norm(ce) + 1e-12)
        r = float((cg @ ce) / denom)
        r2 = r * r
        out[int(d)] = float(1.0 - r2)
    return out


# ---------------------------
# Local PCA dimension variability (curvature proxy)
# ---------------------------

def local_pca_dim_stats(X: np.ndarray, k: int = 12, var_threshold: float = 0.95) -> Dict[str, float]:
    """
    For each point: do PCA on its k-NN neighborhood, record smallest local dimension
    needed to explain var_threshold variance. Then summarize.
    """
    X = np.asarray(X, float)
    n = len(X)
    if n < max(20, k + 2):
        return {"local_dim_mean": float("nan"), "local_dim_std": float("nan"),
                "local_dim_min": float("nan"), "local_dim_max": float("nan")}

    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
    _, idx = nbrs.kneighbors(X)
    dims = []
    for i in range(n):
        nb = X[idx[i, 1:]]  # exclude self
        p = PCA().fit(nb)
        cum = np.cumsum(p.explained_variance_ratio_)
        d = int(np.searchsorted(cum, var_threshold) + 1)
        dims.append(d)

    dims = np.array(dims, int)
    return {
        "local_dim_mean": float(dims.mean()),
        "local_dim_std": float(dims.std()),
        "local_dim_min": float(dims.min()),
        "local_dim_max": float(dims.max()),
    }


# ---------------------------
# Nulls
# ---------------------------

def gaussian_null(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mu, C, _L = stable_cov(X, eps_scale=1e-8)
    return rng.multivariate_normal(mu, C, size=len(X))

def column_shuffle_null(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xn = np.asarray(X, float).copy()
    for j in range(Xn.shape[1]):
        Xn[:, j] = Xn[rng.permutation(len(Xn)), j]
    return Xn


# ---------------------------
# Data loading
# ---------------------------

def load_continuous_lens_space(lens_root: Path) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Loads continuous lens vectors from:
      lens_root/<obs>/lenses_continuous.csv
    Returns:
      X_all: (N,10)
      used: list of observables used
      per_obs: dict obs -> (Ni,10)
    """
    subdirs = [d for d in lens_root.iterdir() if d.is_dir()]
    blocks = []
    used = []
    per_obs = {}

    for d in sorted(subdirs):
        f = d / "lenses_continuous.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        num_cols = [c for c in num_cols if c.lower() not in ("bin", "index")]
        if len(num_cols) < 10:
            continue
        X = df[num_cols[:10]].values.astype(float)
        blocks.append(X)
        used.append(d.name)
        per_obs[d.name] = X

    if not blocks:
        raise RuntimeError(f"No usable lenses_continuous.csv found under: {lens_root}")

    X_all = np.vstack(blocks)
    return X_all, used, per_obs


# ---------------------------
# Report bundle
# ---------------------------

@dataclass
class BlockResult:
    name: str
    n: int
    d: int
    H1: float
    H2: float
    TwoNN: float
    MLE: float
    PCA_cum3: float
    PCA_evr: List[float]
    dmaps_eigs: List[float]
    isomap_resid: Dict[int, float]
    local_dim: Dict[str, float]


def analyze_block(name: str, X: np.ndarray, maxdim: int, isomap_dims: List[int]) -> BlockResult:
    X = np.asarray(X, float)
    evr, _ = pca_spectrum(X)
    pca_cum3 = float(np.sum(evr[:3])) if len(evr) >= 3 else float("nan")
    H0, H1, H2 = compute_ph_max(X, maxdim=maxdim)
    tw = twonn_dimension(X)
    mle = levina_bickel_mle_dimension(X, k=10)
    dme = diffusion_maps_eigs(X, n_eigs=10, sigma=None)
    iso = isomap_residual_curve(X, dims=isomap_dims, n_neighbors=10)
    loc = local_pca_dim_stats(X, k=12, var_threshold=0.95)

    return BlockResult(
        name=name,
        n=int(len(X)),
        d=int(X.shape[1]),
        H1=float(H1),
        H2=float(H2),
        TwoNN=float(tw),
        MLE=float(mle),
        PCA_cum3=float(pca_cum3),
        PCA_evr=[float(x) for x in evr[:10]],
        dmaps_eigs=[float(x) for x in dme],
        isomap_resid={int(k): float(v) for k, v in iso.items()},
        local_dim={k: float(v) for k, v in loc.items()},
    )


def null_stats(name: str, X_real: np.ndarray, make_null, reps: int, rng, maxdim: int) -> Dict[str, float]:
    H1s, H2s, tws, mles, cum3s = [], [], [], [], []
    for _ in range(reps):
        Xn = make_null(X_real)
        H0, H1, H2 = compute_ph_max(Xn, maxdim=maxdim)
        tws.append(twonn_dimension(Xn))
        mles.append(levina_bickel_mle_dimension(Xn, k=10))
        evr, _ = pca_spectrum(Xn)
        cum3s.append(float(np.sum(evr[:3])) if len(evr) >= 3 else float("nan"))
        H1s.append(H1); H2s.append(H2)

    def pack(a):
        a = np.array(a, float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    H1m, H1s_ = pack(H1s)
    H2m, H2s_ = pack(H2s)
    twm, tws_ = pack(tws)
    mlem, mles_ = pack(mles)
    c3m, c3s_ = pack(cum3s)

    return {
        "name": name,
        "H1_mean": H1m, "H1_std": H1s_,
        "H2_mean": H2m, "H2_std": H2s_,
        "TwoNN_mean": twm, "TwoNN_std": tws_,
        "MLE_mean": mlem, "MLE_std": mles_,
        "PCA_cum3_mean": c3m, "PCA_cum3_std": c3s_,
    }


def zscore(x: float, mean: float, std: float) -> float:
    return float((x - mean) / (std + 1e-12))


# ---------------------------
# Plotting (matplotlib only)
# ---------------------------

def plot_pca_evr(evr: np.ndarray, out: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(evr) + 1), evr, marker="o")
    plt.xlabel("component")
    plt.ylabel("explained variance ratio")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def plot_dmaps_eigs(eigs: List[float], out: Path, title: str):
    import matplotlib.pyplot as plt
    if len(eigs) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(eigs) + 1), eigs, marker="o")
    plt.xlabel("eigen index")
    plt.ylabel("eigenvalue")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def plot_isomap_curve(resid: Dict[int, float], out: Path, title: str):
    import matplotlib.pyplot as plt
    if not resid:
        return
    xs = sorted(resid.keys())
    ys = [resid[k] for k in xs]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("target dimension")
    plt.ylabel("residual variance (1 - R^2)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def plot_local_dim_hist(local_dims: List[int], out: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.hist(local_dims, bins=np.arange(min(local_dims), max(local_dims) + 2) - 0.5)
    plt.xlabel("local PCA dimension (95% var)")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-root", default="hepdata_to_dashi", help="Root containing */lenses_continuous.csv")
    ap.add_argument("--out", default="hepdata_manifold_report", help="Output directory")
    ap.add_argument("--maxdim", type=int, default=2, help="PH max homology dim")
    ap.add_argument("--null-reps", type=int, default=50, help="Null repetitions")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--per-observable", action="store_true", help="Also analyze each observable separately")
    args = ap.parse_args()

    lens_root = Path(args.lens_root)
    out = Path(args.out)
    ensure_dir(out)

    rng = np.random.default_rng(args.seed)

    X, used, per_obs = load_continuous_lens_space(lens_root)
    print(f"\nLoaded continuous lens space from: {lens_root}")
    print(f"Observables used: {used}")
    print(f"Total points: {len(X)}  Dimension: {X.shape[1]}")

    isomap_dims = [1, 2, 3, 4, 5, 6]

    # Real blocks
    res_raw = analyze_block("REAL_raw", X, maxdim=args.maxdim, isomap_dims=isomap_dims)
    Xw = whiten_mahalanobis(X)
    res_wh = analyze_block("REAL_whitened", Xw, maxdim=args.maxdim, isomap_dims=isomap_dims)

    # Nulls in raw space
    null_gauss_raw = null_stats("NULL_gaussian_raw", X, lambda Xr: gaussian_null(Xr, rng), args.null_reps, rng, args.maxdim)
    null_shuf_raw = null_stats("NULL_shuffle_raw", X, lambda Xr: column_shuffle_null(Xr, rng), args.null_reps, rng, args.maxdim)

    # Fixed whitening map based on REAL X (so comparisons are consistent)
    mu, C, L = stable_cov(X, eps_scale=1e-6)

    def null_gauss_wh():
        Xn = gaussian_null(X, rng)
        return whiten_with_map(Xn, mu, L)

    def null_shuf_wh():
        Xn = column_shuffle_null(X, rng)
        return whiten_with_map(Xn, mu, L)

    null_gauss_wh = null_stats("NULL_gaussian_then_whiten", X, lambda Xr: null_gauss_wh(), args.null_reps, rng, args.maxdim)
    null_shuf_wh = null_stats("NULL_shuffle_then_whiten", X, lambda Xr: null_shuf_wh(), args.null_reps, rng, args.maxdim)

    # Subspace: top variance 4D (report only; you can change later)
    var = np.var(X, axis=0)
    idx4 = np.argsort(var)[::-1][:4]
    X4 = X[:, idx4]
    res_sub = analyze_block(f"REAL_topvar4_idx={idx4.tolist()}", X4, maxdim=args.maxdim, isomap_dims=isomap_dims)

    # Z-scores (headline)
    z = {
        "RAW_H1_vs_gauss": zscore(res_raw.H1, null_gauss_raw["H1_mean"], null_gauss_raw["H1_std"]),
        "RAW_TwoNN_vs_gauss": zscore(res_raw.TwoNN, null_gauss_raw["TwoNN_mean"], null_gauss_raw["TwoNN_std"]),
        "WH_H1_vs_gauss": zscore(res_wh.H1, null_gauss_wh["H1_mean"], null_gauss_wh["H1_std"]),
        "WH_TwoNN_vs_gauss": zscore(res_wh.TwoNN, null_gauss_wh["TwoNN_mean"], null_gauss_wh["TwoNN_std"]),
    }

    # Save report
    report = {
        "meta": {
            "lens_root": str(lens_root),
            "observables_used": used,
            "total_points": int(len(X)),
            "ambient_dim": int(X.shape[1]),
            "seed": int(args.seed),
            "null_reps": int(args.null_reps),
        },
        "real": {
            "raw": res_raw.__dict__,
            "whitened": res_wh.__dict__,
            "topvar4": res_sub.__dict__,
        },
        "nulls": {
            "gaussian_raw": null_gauss_raw,
            "shuffle_raw": null_shuf_raw,
            "gaussian_then_whiten": null_gauss_wh,
            "shuffle_then_whiten": null_shuf_wh,
        },
        "z_scores": z,
    }

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Summary CSV
    rows = []
    for r in [res_raw, res_wh, res_sub]:
        rows.append({
            "name": r.name,
            "n": r.n,
            "d": r.d,
            "H1": r.H1,
            "H2": r.H2,
            "TwoNN": r.TwoNN,
            "MLE": r.MLE,
            "PCA_cum3": r.PCA_cum3,
            "local_dim_mean": r.local_dim.get("local_dim_mean", np.nan),
            "local_dim_std": r.local_dim.get("local_dim_std", np.nan),
        })
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    # Plots
    evr_raw, _ = pca_spectrum(X)
    plot_pca_evr(evr_raw[:10], out / "pca_evr_raw.png", "PCA explained variance ratio (RAW)")

    evr_wh, _ = pca_spectrum(Xw)
    plot_pca_evr(evr_wh[:10], out / "pca_evr_whitened.png", "PCA explained variance ratio (WHITENED)")

    plot_dmaps_eigs(res_raw.dmaps_eigs, out / "diffusion_eigs_raw.png", "Diffusion map eigenvalues (RAW)")
    plot_dmaps_eigs(res_wh.dmaps_eigs, out / "diffusion_eigs_whitened.png", "Diffusion map eigenvalues (WHITENED)")

    plot_isomap_curve(res_raw.isomap_resid, out / "isomap_resid_raw.png", "Isomap residual variance curve (RAW)")
    plot_isomap_curve(res_wh.isomap_resid, out / "isomap_resid_whitened.png", "Isomap residual variance curve (WHITENED)")

    # Optional per-observable breakdown
    if args.per_observable:
        per_dir = out / "per_observable"
        ensure_dir(per_dir)
        per_rows = []
        for obs, Xo in per_obs.items():
            r = analyze_block(f"{obs}_raw", Xo, maxdim=args.maxdim, isomap_dims=isomap_dims)
            per_rows.append({
                "obs": obs,
                "n": r.n,
                "H1": r.H1,
                "H2": r.H2,
                "TwoNN": r.TwoNN,
                "MLE": r.MLE,
                "PCA_cum3": r.PCA_cum3,
            })
        pd.DataFrame(per_rows).to_csv(per_dir / "per_observable_summary.csv", index=False)

    # Console headline
    print("\n=== MANIFOLD REPORT (headline) ===")
    print(f"RAW:    TwoNN={res_raw.TwoNN:.3f}  MLE={res_raw.MLE:.3f}  PCA_cum3={res_raw.PCA_cum3:.3f}  H1={res_raw.H1:.4f}  H2={res_raw.H2:.4f}")
    print(f"WHITE:  TwoNN={res_wh.TwoNN:.3f}  MLE={res_wh.MLE:.3f}  PCA_cum3={res_wh.PCA_cum3:.3f}  H1={res_wh.H1:.4f}  H2={res_wh.H2:.4f}")
    print(f"Null(Gauss RAW): TwoNN={null_gauss_raw['TwoNN_mean']:.3f}±{null_gauss_raw['TwoNN_std']:.3f}  H1={null_gauss_raw['H1_mean']:.4f}±{null_gauss_raw['H1_std']:.4f}")
    print(f"Z: RAW TwoNN vs Gauss = {z['RAW_TwoNN_vs_gauss']:.2f}σ ; RAW H1 vs Gauss = {z['RAW_H1_vs_gauss']:.2f}σ")
    print(f"Outputs written to: {out.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
