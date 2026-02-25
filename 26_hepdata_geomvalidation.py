#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ---------------------------
# Persistent homology helpers
# ---------------------------

def max_persistence(dgm):
    if dgm is None or len(dgm) == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[np.isfinite(pers)]
    if len(pers) == 0:
        return 0.0
    return float(np.max(pers))


def compute_ph(X, maxdim=2, distance_matrix=False):
    out = ripser(X, maxdim=maxdim, distance_matrix=distance_matrix)
    dgms = out["dgms"]
    H0 = max_persistence(dgms[0])
    H1 = max_persistence(dgms[1]) if len(dgms) > 1 else 0.0
    H2 = max_persistence(dgms[2]) if len(dgms) > 2 else 0.0
    return H0, H1, H2, dgms


# ---------------------------
# TwoNN intrinsic dimension
# ---------------------------

def twonn_dimension(X):
    """
    Facco et al. TwoNN estimator:
      m = 1 / mean(log(r2/r1))
    using nearest neighbors 1 and 2.
    """
    X = np.asarray(X, float)
    if len(X) < 5:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)

    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mu = r2 / (r1 + 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 0)]
    if len(mu) == 0:
        return float("nan")
    return float(1.0 / (np.mean(np.log(mu)) + 1e-12))


# ---------------------------
# Metrics: whitening / nulls
# ---------------------------

def whiten_mahalanobis(X, eps=1e-6):
    """
    Whitening such that Euclidean distances in Xw correspond roughly
    to Mahalanobis distances in X.
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    C = np.cov(X.T)
    C = 0.5 * (C + C.T)
    C = C + eps * np.eye(C.shape[0])
    L = np.linalg.cholesky(C)
    Xc = X - mu
    Xw = np.linalg.solve(L, Xc.T).T
    return Xw


def gaussian_null(X, rng):
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    C = np.cov(X.T)
    C = 0.5 * (C + C.T)
    # small jitter for numerical stability
    C = C + 1e-8 * np.trace(C) / max(1, C.shape[0]) * np.eye(C.shape[0])
    return rng.multivariate_normal(mu, C, size=len(X))


def column_shuffle_null(X, rng):
    X = np.asarray(X, float)
    Xn = X.copy()
    for j in range(Xn.shape[1]):
        perm = rng.permutation(len(Xn))
        Xn[:, j] = Xn[perm, j]
    return Xn


# ---------------------------
# Loading continuous lens space
# ---------------------------

def load_continuous_lens_space(root_dir):
    """
    Expects:
      hepdata_to_dashi/<observable>/lenses_continuous.csv
    where file has columns: bin, L0..L9 (or similar numeric columns).
    """
    root = Path(root_dir)
    if not root.exists():
        raise RuntimeError(f"Lens root dir not found: {root_dir}")

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No subdirectories in: {root_dir}")

    blocks = []
    used = []

    for d in sorted(subdirs):
        f = d / "lenses_continuous.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        num_cols = list(df.select_dtypes(include=[np.number]).columns)

        # drop bin if present
        num_cols = [c for c in num_cols if c.lower() not in ("bin", "index")]

        if len(num_cols) < 10:
            continue

        X = df[num_cols[:10]].values.astype(float)
        blocks.append(X)
        used.append(d.name)

    if not blocks:
        raise RuntimeError("No lenses_continuous.csv files found (or no 10 numeric lens columns).")

    X = np.vstack(blocks)
    return X, used


# ---------------------------
# Residual / active subspace
# ---------------------------

def top_variance_subspace(X, k=4):
    var = np.var(X, axis=0)
    idx = np.argsort(var)[::-1][:k]
    return X[:, idx], idx


# ---------------------------
# PCA spectrum
# ---------------------------

def pca_spectrum(X):
    pca = PCA().fit(X)
    return pca.explained_variance_ratio_, pca.singular_values_


# ---------------------------
# Contraction coeff-flow (from hepdata_dashi_native metrics)
# ---------------------------

def load_coeff_trajectories(metrics_root):
    """
    Looks for:
      hepdata_dashi_native/*_dashi_native_metrics.csv
    and extracts columns:
      alpha, b0..b4

    Returns dict[label] = (alphas_sorted, betas_sorted)
    """
    root = Path(metrics_root)
    if not root.exists():
        return {}

    files = sorted(root.glob("*_dashi_native_metrics.csv"))
    trajs = {}

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        needed = ["alpha", "b0", "b1", "b2", "b3", "b4"]
        if not all(c in df.columns for c in needed):
            continue

        alphas = df["alpha"].values.astype(float)
        betas = df[["b0", "b1", "b2", "b3", "b4"]].values.astype(float)

        # sort by alpha (increasing)
        order = np.argsort(alphas)
        alphas = alphas[order]
        betas = betas[order]

        label = f.name.replace("_dashi_native_metrics.csv", "")
        trajs[label] = (alphas, betas)

    return trajs


def per_step_ph_over_flow(alphas, betas, maxdim=2):
    """
    Memory-safe: compute PH on the *prefix* of points up to each step.
    This is how topology builds along the RG/projection strength axis.
    """
    rows = []
    for i in range(3, len(betas) + 1):
        Xp = betas[:i]
        _, H1, H2, _ = compute_ph(Xp, maxdim=maxdim)
        rows.append((alphas[i-1], i, H1, H2))
    return rows


# ---------------------------
# Reporting
# ---------------------------

def summarize_ph(name, X, maxdim=2):
    _, H1, H2, _ = compute_ph(X, maxdim=maxdim)
    dim = twonn_dimension(X)
    evr, _ = pca_spectrum(X)
    return {
        "name": name,
        "n": int(len(X)),
        "d": int(X.shape[1]),
        "H1": float(H1),
        "H2": float(H2),
        "TwoNN_dim": float(dim),
        "PCA_evr_1": float(evr[0]) if len(evr) > 0 else float("nan"),
        "PCA_evr_2": float(evr[1]) if len(evr) > 1 else float("nan"),
        "PCA_evr_3": float(evr[2]) if len(evr) > 2 else float("nan"),
        "PCA_evr_cum3": float(np.sum(evr[:3])) if len(evr) >= 3 else float("nan"),
    }


def print_summary(s):
    print(f"\n=== {s['name']} ===")
    print(f"n={s['n']}  d={s['d']}")
    print(f"H1_max={s['H1']:.4f}  H2_max={s['H2']:.4f}")
    print(f"TwoNN_dim≈{s['TwoNN_dim']:.3f}")
    print(f"PCA EVR: [{s['PCA_evr_1']:.3f}, {s['PCA_evr_2']:.3f}, {s['PCA_evr_3']:.3f}]  cum3={s['PCA_evr_cum3']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-root", default="hepdata_to_dashi", help="Root directory containing */lenses_continuous.csv")
    ap.add_argument("--metrics-root", default="hepdata_dashi_native", help="Root directory containing *_dashi_native_metrics.csv")
    ap.add_argument("--maxdim", type=int, default=2, help="Max homology dim for ripser (2 recommended)")
    ap.add_argument("--null-reps", type=int, default=20, help="Number of null repetitions")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--subspace-k", type=int, default=4, help="Top-variance subspace dimension")
    ap.add_argument("--save-csv", action="store_true", help="Save summary CSVs into ./hepdata_geometry_validation")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    outdir = Path("hepdata_geometry_validation")
    if args.save_csv:
        outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load continuous lens points
    # ---------------------------
    X, used_obs = load_continuous_lens_space(args.lens_root)
    print(f"\nLoaded continuous lens space from: {args.lens_root}")
    print(f"Observables used: {used_obs}")
    print(f"Total points: {len(X)}  Dimension: {X.shape[1]}")

    # ---------------------------
    # Option A: PH on continuous lens vectors (raw vs whitened)
    # ---------------------------
    s_raw = summarize_ph("Continuous Lens Space (Raw Euclidean)", X, maxdim=args.maxdim)
    print_summary(s_raw)

    Xw = whiten_mahalanobis(X)
    s_wh = summarize_ph("Continuous Lens Space (Whitened ≈ Mahalanobis)", Xw, maxdim=args.maxdim)
    print_summary(s_wh)

    # ---------------------------
    # Nulls (Gaussian + column shuffle), under both metrics
    # ---------------------------
    def null_block(name, Xbase, make_null):
        H1s, H2s, dims, cum3s = [], [], [], []
        for _ in range(args.null_reps):
            Xn = make_null(Xbase)
            _, H1, H2, _ = compute_ph(Xn, maxdim=args.maxdim)
            H1s.append(H1); H2s.append(H2)
            dims.append(twonn_dimension(Xn))
            evr, _ = pca_spectrum(Xn)
            cum3s.append(float(np.sum(evr[:3])) if len(evr) >= 3 else float("nan"))
        H1s = np.array(H1s); H2s = np.array(H2s); dims = np.array(dims); cum3s = np.array(cum3s)
        print(f"\n--- Null: {name} (reps={args.null_reps}) ---")
        print(f"H1 mean={H1s.mean():.4f}  std={H1s.std():.4f}  min={H1s.min():.4f}  max={H1s.max():.4f}")
        print(f"H2 mean={H2s.mean():.4f}  std={H2s.std():.4f}  min={H2s.min():.4f}  max={H2s.max():.4f}")
        print(f"TwoNN dim mean={np.nanmean(dims):.3f}  std={np.nanstd(dims):.3f}")
        print(f"PCA cum3 mean={np.nanmean(cum3s):.3f}  std={np.nanstd(cum3s):.3f}")
        return H1s, H2s, dims, cum3s

    # raw metric nulls
    null_block("Gaussian(same cov) on RAW", X, lambda Xb: gaussian_null(Xb, rng))
    null_block("Column-shuffle on RAW", X, lambda Xb: column_shuffle_null(Xb, rng))

    # whitened metric nulls: generate null in raw, then whiten using the same whitening map as real
    # (to make the comparison consistent)
    mu = X.mean(axis=0)
    C = np.cov(X.T); C = 0.5*(C+C.T) + 1e-6*np.eye(X.shape[1])
    L = np.linalg.cholesky(C)

    def whiten_with_fixed_map(Xn):
        return np.linalg.solve(L, (Xn - mu).T).T

    null_block("Gaussian(same cov) then WHITEN", X, lambda Xb: whiten_with_fixed_map(gaussian_null(Xb, rng)))
    null_block("Column-shuffle then WHITEN", X, lambda Xb: whiten_with_fixed_map(column_shuffle_null(Xb, rng)))

    # ---------------------------
    # Option B: Residual/active subspace PH (top variance lenses)
    # ---------------------------
    Xsub, idx = top_variance_subspace(X, k=args.subspace_k)
    s_sub = summarize_ph(f"Top-Variance Subspace (k={args.subspace_k}, idx={idx.tolist()})", Xsub, maxdim=args.maxdim)
    print_summary(s_sub)

    Xsub_w = whiten_mahalanobis(Xsub)
    s_subw = summarize_ph(f"Top-Variance Subspace WHITENED (k={args.subspace_k})", Xsub_w, maxdim=args.maxdim)
    print_summary(s_subw)

    # ---------------------------
    # Option C: Contraction flow in coefficient space (from your existing metrics CSVs)
    # ---------------------------
    trajs = load_coeff_trajectories(args.metrics_root)
    if not trajs:
        print(f"\n(No coefficient-flow files found in: {args.metrics_root} — skipping contraction-flow PH.)")
    else:
        print(f"\nLoaded contraction coefficient flows from: {args.metrics_root}")
        flow_rows_all = []

        for label, (alphas, betas) in trajs.items():
            # PH on the set of beta points (small N)
            _, H1b, H2b, _ = compute_ph(betas, maxdim=min(args.maxdim, 2))
            print(f"\n=== Coefficient Flow (beta space): {label} ===")
            print(f"points={len(betas)}  dim={betas.shape[1]}")
            print(f"PH on beta points: H1_max={H1b:.4f}  H2_max={H2b:.4f}")

            # Per-step PH over prefix (memory-safe)
            rows = per_step_ph_over_flow(alphas, betas, maxdim=min(args.maxdim, 2))
            # print a compact view (alpha, H1, H2)
            for (a, npts, h1, h2) in rows:
                flow_rows_all.append((label, a, npts, h1, h2))

            # show last few steps (most projected)
            tail = rows[-5:] if len(rows) >= 5 else rows
            print("Last steps (alpha, npts, H1, H2):")
            for a, npts, h1, h2 in tail:
                print(f"  alpha={a:.3e}  n={npts:2d}  H1={h1:.4f}  H2={h2:.4f}")

        if args.save_csv and flow_rows_all:
            df_flow = pd.DataFrame(flow_rows_all, columns=["label", "alpha", "npts", "H1_max", "H2_max"])
            df_flow.to_csv(outdir / "contraction_flow_ph.csv", index=False)
            print(f"\nSaved: {outdir / 'contraction_flow_ph.csv'}")

    # ---------------------------
    # Save summaries
    # ---------------------------
    if args.save_csv:
        df = pd.DataFrame([s_raw, s_wh, s_sub, s_subw])
        df.to_csv(outdir / "geometry_summary.csv", index=False)
        print(f"\nSaved: {outdir / 'geometry_summary.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
