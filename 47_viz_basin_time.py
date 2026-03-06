#!/usr/bin/env python3
"""
47_viz_basin_time.py

Basin / time-to-converge visualization in PCA embedding.
Colors each state by steps-to-final within its label trajectory.
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
    ap.add_argument("--out", default="viz_basin_time.png")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    for c in ["label", "iter"] + args.cols:
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")

    X = df[args.cols].values.astype(float)
    Z = compute_pca(X, n=2)
    df["z1"] = Z[:, 0]
    df["z2"] = Z[:, 1]

    # steps-to-final per label
    steps_to_final = []
    for lab, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        n = len(g)
        for i in range(n):
            steps_to_final.append(n - 1 - i)
    df["t2f"] = steps_to_final

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["z1"], df["z2"], c=df["t2f"], cmap="viridis", s=10)
    ax.set_title("Time-to-Converge (steps to final)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="steps to final")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
