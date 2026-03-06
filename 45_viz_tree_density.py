#!/usr/bin/env python3
"""
45_viz_tree_density.py

Tree density map (prefix tree) from ternary lens states.
Produces an icicle plot: prefix depth on x-axis, nodes stacked by count.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_ternary(root: Path) -> list[np.ndarray]:
    rows = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        f = d / "lenses_ternary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("L")]
        X = df[cols].values.astype(int)
        for i in range(len(X)):
            rows.append(X[i])
    return rows


def build_prefix_counts(X: list[np.ndarray], depth: int) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for x in X:
        prefix = tuple(x[:depth])
        counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="hepdata_to_dashi_all")
    ap.add_argument("--depth", type=int, default=6, help="prefix depth")
    ap.add_argument("--out", default="viz_tree_density.png")
    args = ap.parse_args()

    X = load_ternary(Path(args.root))
    if not X:
        raise SystemExit("no ternary data found")

    depth = args.depth
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"Prefix Tree Density (depth={depth})")
    ax.set_xlabel("prefix depth")
    ax.set_ylabel("count")
    ax.set_xticks(range(depth + 1))

    # build icicle by depth
    total = len(X)
    y0 = 0
    for d in range(1, depth + 1):
        counts = build_prefix_counts(X, d)
        # stack rectangles by count
        y = 0
        for prefix, cnt in sorted(counts.items()):
            ax.add_patch(plt.Rectangle((d - 1, y), 1.0, cnt, alpha=0.6))
            y += cnt
        y0 = max(y0, y)

    ax.set_ylim(0, y0)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
