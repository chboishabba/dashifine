#!/usr/bin/env python3
"""
43_viz_ultrametric_tree.py

Ultrametric geometry visualization using LCP-depth distance.
Computes pairwise distances for ternary lens states and embeds with MDS.
Outputs an animated GIF showing points added in bin order per label.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.manifold import MDS


def lcp_depth(a: np.ndarray, b: np.ndarray) -> int:
    diff = a != b
    if not np.any(diff):
        return int(a.shape[0])
    return int(np.argmax(diff))


def ultrametric_distance_matrix(X: np.ndarray, base: float = 2.0) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = base ** (-float(lcp_depth(X[i], X[j])))
            D[i, j] = d
            D[j, i] = d
    return D


def load_ternary(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    labels = []
    bins = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        f = d / "lenses_ternary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("L")]
        X = df[cols].values.astype(float)
        for i in range(len(X)):
            rows.append(X[i])
            labels.append(d.name)
            bins.append(int(df.iloc[i]["bin"]) if "bin" in df.columns else i)
    return np.array(rows), np.array(labels), np.array(bins)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="hepdata_to_dashi_all")
    ap.add_argument("--out", default="viz_ultrametric_tree.gif")
    ap.add_argument("--base", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)
    X, labels, bins = load_ternary(root)
    if X.size == 0:
        raise SystemExit("no ternary data found")

    D = ultrametric_distance_matrix(X, base=args.base)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    Y = mds.fit_transform(D)

    unique_labels = sorted(set(labels))
    colors = {lab: plt.cm.tab20(i % 20) for i, lab in enumerate(unique_labels)}

    # order by bin (within label)
    order = np.lexsort((bins, labels))
    Y = Y[order]
    labels = labels[order]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Ultrametric Geometry (MDS)")
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")

    # fix axis limits up front (important for blitting)
    x_min, x_max = float(np.min(Y[:, 0])), float(np.max(Y[:, 0]))
    y_min, y_max = float(np.min(Y[:, 1])), float(np.max(Y[:, 1]))
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # initialize with first point to avoid empty scatter in blit
    scat = ax.scatter([Y[0, 0]], [Y[0, 1]], s=10, c=[colors[labels[0]]])

    def update(frame):
        n = min(frame + 1, len(Y))
        pts = Y[:n]
        cols = [colors[l] for l in labels[:n]]
        scat.set_offsets(pts)
        scat.set_color(cols)
        ax.set_title(f"Ultrametric Geometry (MDS)  n={n}")
        return scat,

    total_frames = min(len(Y), args.max_frames)
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000//args.fps, blit=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
