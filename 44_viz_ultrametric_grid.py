#!/usr/bin/env python3
"""
44_viz_ultrametric_grid.py

Combined 2x2 grid visualization on ultrametric MDS embedding:
  - Shell (|Q_sigma|) coloring
  - Orbit ID coloring (signed permutations within two blocks)
  - Bin index coloring
  - Attractor cluster coloring (from closure embedding final states)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.manifold import MDS
from sklearn.cluster import KMeans


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


def load_ternary(root: Path):
    rows = []
    labels = []
    bins = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        f = d / "lenses_ternary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("L")]
        X = df[cols].values.astype(int)
        for i in range(len(X)):
            rows.append(X[i])
            labels.append(d.name)
            bins.append(int(df.iloc[i]["bin"]) if "bin" in df.columns else i)
    return np.array(rows), np.array(labels), np.array(bins)


def shell_value(x: np.ndarray, split: int) -> int:
    # Q_sigma = (#nonzero in +block) - (#nonzero in -block)
    a = np.sum(x[:split] != 0)
    b = np.sum(x[split:] != 0)
    return abs(int(a - b))


def orbit_id(x: np.ndarray, split: int) -> tuple:
    # signed permutations within blocks ⇒ orbit determined by sorted abs values per block
    a = tuple(sorted(np.abs(x[:split]).tolist()))
    b = tuple(sorted(np.abs(x[split:]).tolist()))
    return (a, b)


def label_attractor_clusters(closure_csv: Path, cols: list[str], k: int):
    df = pd.read_csv(closure_csv)
    for c in ["label", "iter"] + cols:
        if c not in df.columns:
            raise SystemExit(f"missing column in closure embedding: {c}")
    # final point per label
    finals = []
    labels = []
    for lab, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        finals.append(g[cols].iloc[-1].values.astype(float))
        labels.append(lab)
    finals = np.vstack(finals)
    k = min(k, len(labels))
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    cl = km.fit_predict(finals)
    return {lab: int(c) for lab, c in zip(labels, cl)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="hepdata_to_dashi_all")
    ap.add_argument("--closure", default="hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv")
    ap.add_argument("--closure-cols", nargs="+", default=["v_pnorm", "v_dnorm", "v_arrow"])
    ap.add_argument("--split", type=int, default=5, help="block split index for signed-block action")
    ap.add_argument("--base", type=float, default=2.0)
    ap.add_argument("--out", default="viz_ultrametric_grid.gif")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--clusters", type=int, default=6, help="attractor cluster count")
    args = ap.parse_args()

    X, labels, bins = load_ternary(Path(args.root))
    if X.size == 0:
        raise SystemExit("no ternary data found")

    # compute embedding
    D = ultrametric_distance_matrix(X, base=args.base)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    Y = mds.fit_transform(D)

    # ordering by label, then bin
    order = np.lexsort((bins, labels))
    X = X[order]
    labels = labels[order]
    bins = bins[order]
    Y = Y[order]

    # colors: shell
    shell_vals = np.array([shell_value(x, args.split) for x in X])
    shell_norm = (shell_vals - shell_vals.min()) / (shell_vals.max() - shell_vals.min() + 1e-9)
    shell_colors = plt.cm.viridis(shell_norm)

    # colors: orbit id (hash to palette)
    orbit_ids = [orbit_id(x, args.split) for x in X]
    orbit_map = {}
    orbit_idx = []
    for oid in orbit_ids:
        if oid not in orbit_map:
            orbit_map[oid] = len(orbit_map)
        orbit_idx.append(orbit_map[oid])
    orbit_idx = np.array(orbit_idx)
    orbit_colors = plt.cm.tab20((orbit_idx % 20) / 20.0)

    # colors: bin index
    bnorm = (bins - bins.min()) / (bins.max() - bins.min() + 1e-9)
    bin_colors = plt.cm.plasma(bnorm)

    # colors: attractor clusters (by label)
    cl_map = label_attractor_clusters(Path(args.closure), args.closure_cols, args.clusters)
    cl_idx = np.array([cl_map.get(l, 0) for l in labels])
    cl_colors = plt.cm.Set2((cl_idx % 8) / 8.0)

    # setup figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    titles = ["Shell |Q|", "Orbit ID (signed-block)", "Bin index", "Attractor cluster"]
    scatters = []

    # axis limits
    x_min, x_max = float(np.min(Y[:, 0])), float(np.max(Y[:, 0]))
    y_min, y_max = float(np.min(Y[:, 1])), float(np.max(Y[:, 1]))
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)

    for ax, title in zip(axes.ravel(), titles):
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        # init with first point
        scat = ax.scatter([Y[0, 0]], [Y[0, 1]], s=10)
        scatters.append(scat)

    def update(frame):
        n = min(frame + 1, len(Y))
        pts = Y[:n]
        scatters[0].set_offsets(pts)
        scatters[0].set_color(shell_colors[:n])
        scatters[1].set_offsets(pts)
        scatters[1].set_color(orbit_colors[:n])
        scatters[2].set_offsets(pts)
        scatters[2].set_color(bin_colors[:n])
        scatters[3].set_offsets(pts)
        scatters[3].set_color(cl_colors[:n])
        return scatters

    total_frames = min(len(Y), args.max_frames)
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000//args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
