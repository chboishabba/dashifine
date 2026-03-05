#!/usr/bin/env python3
"""
41_viz_cone_monotone.py

Animated GIF of Q(Δs) vs iteration for each label (cone monotonicity).
Uses G_mask3.npy by default and closure embedding per-step CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def compute_deltas(df: pd.DataFrame, shape_cols: list[str]) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        for i in range(len(g) - 1):
            r0 = g.iloc[i]
            r1 = g.iloc[i + 1]
            row = {
                "label": label,
                "iter_from": int(r0["iter"]),
                "iter_to": int(r1["iter"]),
            }
            for c in shape_cols:
                row[f"d_{c}"] = float(r1[c] - r0[c])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv")
    ap.add_argument("--shape-cols", nargs="+", default=["v_pnorm", "v_dnorm", "v_arrow"])
    ap.add_argument("--G", default="G_mask3.npy")
    ap.add_argument("--out", default="viz_cone_monotone.gif")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    for c in ["label", "iter"] + args.shape_cols:
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")

    ddf = compute_deltas(df, args.shape_cols)
    G = np.load(args.G)
    if G.shape[0] != len(args.shape_cols):
        raise SystemExit("G dimension does not match shape-cols length")

    # compute Q per step
    d_shape = np.vstack([ddf[f"d_{c}"].values for c in args.shape_cols]).T
    q_vals = np.einsum("bi,ij,bj->b", d_shape, G, d_shape)
    ddf["Q"] = q_vals

    labels = sorted(ddf["label"].unique())
    label_colors = {lab: plt.cm.tab20(i % 20) for i, lab in enumerate(labels)}

    # precompute per-label series
    series = {}
    max_len = 0
    for lab in labels:
        g = ddf[ddf["label"] == lab].sort_values("iter_to")
        series[lab] = (g["iter_to"].values, g["Q"].values)
        max_len = max(max_len, len(g))

    # frame count
    total_frames = min(max_len, args.max_frames)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("iter")
    ax.set_ylabel("Q(Δs)")
    ax.set_title("Cone Monotonicity: Q(Δs) vs Iteration")

    lines = {}
    for lab in labels:
        line, = ax.plot([], [], lw=1.5, color=label_colors[lab], alpha=0.9)
        lines[lab] = line

    # axis limits
    all_iters = np.concatenate([series[lab][0] for lab in labels])
    all_q = np.concatenate([series[lab][1] for lab in labels])
    ax.set_xlim(all_iters.min(), all_iters.max())
    q_pad = 0.05 * (all_q.max() - all_q.min() + 1e-9)
    ax.set_ylim(all_q.min() - q_pad, all_q.max() + q_pad)

    def init():
        for lab in labels:
            lines[lab].set_data([], [])
        return list(lines.values())

    def update(frame):
        for lab in labels:
            iters, qs = series[lab]
            n = min(frame + 1, len(iters))
            lines[lab].set_data(iters[:n], qs[:n])
        ax.set_title(f"Cone Monotonicity: Q(Δs) vs Iteration (t={frame+1})")
        return list(lines.values())

    anim = FuncAnimation(fig, update, init_func=init, frames=total_frames, interval=1000//args.fps, blit=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
