#!/usr/bin/env python3
"""
46_viz_density_flow.py

2D density + flow field in PCA embedding space.
Uses closure_embedding_per_step.csv to compute PCA and per-step flow vectors.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_pca(X: np.ndarray, n: int = 2) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :n] * S[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv")
    ap.add_argument("--cols", nargs="+", default=["v_pnorm", "v_dnorm", "v_arrow"])
    ap.add_argument("--out", default="viz_density_flow.png")
    ap.add_argument("--bins", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    for c in ["label", "iter"] + args.cols:
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")

    # PCA embedding for all states
    X = df[args.cols].values.astype(float)
    Z = compute_pca(X, n=2)
    df["z1"] = Z[:, 0]
    df["z2"] = Z[:, 1]

    # per-step flow vectors
    flows = []
    for lab, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        z = g[["z1", "z2"]].values
        for i in range(len(z) - 1):
            flows.append((z[i, 0], z[i, 1], z[i + 1, 0] - z[i, 0], z[i + 1, 1] - z[i, 1]))
    flows = np.array(flows)

    # grid
    x_min, x_max = df["z1"].min(), df["z1"].max()
    y_min, y_max = df["z2"].min(), df["z2"].max()
    nx = ny = args.bins
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    U = np.zeros((ny, nx))
    V = np.zeros((ny, nx))
    C = np.zeros((ny, nx))

    for x, y, dx, dy in flows:
        ix = np.searchsorted(xs, x) - 1
        iy = np.searchsorted(ys, y) - 1
        if 0 <= ix < nx and 0 <= iy < ny:
            U[iy, ix] += dx
            V[iy, ix] += dy
            C[iy, ix] += 1

    # average
    mask = C > 0
    U[mask] /= C[mask]
    V[mask] /= C[mask]

    # density (2D histogram)
    H, xedges, yedges = np.histogram2d(df["z1"], df["z2"], bins=nx)
    H = H.T

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(H, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="Greys", alpha=0.5)
    ax.streamplot(xs, ys, U, V, color="tab:blue", density=1.0, linewidth=1.0)
    ax.set_title("Density + Flow Field (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
