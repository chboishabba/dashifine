#!/usr/bin/env python3
"""
42_viz_closure_flow.py

Closure flow / attractor basins visualization.
- PCA embed of all states from closure_embedding_per_step.csv
- Animate trajectories per label over iterations
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv")
    ap.add_argument("--cols", nargs="+", default=["v_pnorm", "v_dnorm", "v_arrow"])
    ap.add_argument("--out", default="viz_closure_flow.gif")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    for c in ["label", "iter"] + args.cols:
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")

    # PCA to 2D
    X = df[args.cols].values.astype(float)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X2 = U[:, :2] * S[:2]
    df["pc1"] = X2[:, 0]
    df["pc2"] = X2[:, 1]

    labels = sorted(df["label"].unique())
    colors = {lab: plt.cm.tab20(i % 20) for i, lab in enumerate(labels)}

    # per-label trajectory
    series = {}
    max_len = 0
    for lab in labels:
        g = df[df["label"] == lab].sort_values("iter")
        series[lab] = (g["pc1"].values, g["pc2"].values)
        max_len = max(max_len, len(g))

    total_frames = min(max_len, args.max_frames)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Closure Flow (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # plot endpoints as attractors (last point)
    for lab in labels:
        x, y = series[lab]
        ax.scatter(x[-1], y[-1], color=colors[lab], s=20, marker="x", alpha=0.8)

    lines = {}
    for lab in labels:
        line, = ax.plot([], [], lw=1.5, color=colors[lab], alpha=0.9)
        lines[lab] = line

    # axis limits
    ax.set_xlim(df["pc1"].min(), df["pc1"].max())
    ax.set_ylim(df["pc2"].min(), df["pc2"].max())

    def init():
        for lab in labels:
            lines[lab].set_data([], [])
        return list(lines.values())

    def update(frame):
        for lab in labels:
            x, y = series[lab]
            n = min(frame + 1, len(x))
            lines[lab].set_data(x[:n], y[:n])
        ax.set_title(f"Closure Flow (PCA)  t={frame+1}")
        return list(lines.values())

    anim = FuncAnimation(fig, update, init_func=init, frames=total_frames, interval=1000//args.fps, blit=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
