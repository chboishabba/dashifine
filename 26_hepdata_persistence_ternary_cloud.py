#!/usr/bin/env python3
"""
Compute persistent homology of the ternary state cloud.

Each bin is a point in R^10 (ternary {-1,0,1} per lens), pooled across observables.

We compute Vietoris-Rips persistence with ripser (if installed).
If ripser is unavailable, we write an MST-based H0 approximation and a note.

Usage:
  python 26_hepdata_persistence_ternary_cloud.py --inp hepdata_to_dashi --out persistence_out
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_ternary_cloud(inp: Path):
    pts = []
    labels = []
    for obs_dir in sorted([p for p in inp.iterdir() if p.is_dir()]):
        csvp = obs_dir / "lenses_ternary.csv"
        if not csvp.exists():
            continue
        data = np.loadtxt(csvp, delimiter=",", skiprows=1)
        T = data[:, 1:11]
        pts.append(T.astype(float))
        labels += [obs_dir.name] * T.shape[0]
    if not pts:
        raise SystemExit(f"No lenses_ternary.csv found under: {inp}")
    X = np.vstack(pts)
    labels = np.array(labels, dtype=object)
    return X, labels


def plot_diagrams(diagrams, outpath: Path, title: str):
    plt.figure(figsize=(7, 6))
    maxv = 1.0
    for dgm in diagrams:
        if dgm is None or len(dgm) == 0:
            continue
        dgm = np.asarray(dgm, float)
        finite = np.isfinite(dgm[:, 1])
        if finite.any():
            maxv = max(maxv, float(np.max(dgm[finite, 1])))
    plt.plot([0, maxv], [0, maxv], linestyle="--")

    for dim, dgm in enumerate(diagrams):
        if dgm is None or len(dgm) == 0:
            continue
        dgm = np.asarray(dgm, float)
        finite = np.isfinite(dgm[:, 1])
        dgm = dgm[finite]
        if len(dgm) == 0:
            continue
        plt.scatter(dgm[:, 0], dgm[:, 1], s=25, alpha=0.7, label=f"H{dim}")

    plt.title(title)
    plt.xlabel("birth")
    plt.ylabel("death")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def mst_h0_barcode(X: np.ndarray):
    N = X.shape[0]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    in_mst = np.zeros(N, dtype=bool)
    in_mst[0] = True
    death_times = []

    min_dist = D[0].copy()
    min_dist[0] = np.inf
    for _ in range(N - 1):
        j = int(np.argmin(min_dist))
        death_times.append(float(min_dist[j]))
        in_mst[j] = True
        min_dist[j] = np.inf
        min_dist = np.minimum(min_dist, D[j])
        min_dist[in_mst] = np.inf

    dgm0 = np.column_stack([np.zeros(len(death_times)), np.array(death_times)])
    return [dgm0]


def summarize(diagrams):
    out = []
    for dim, dgm in enumerate(diagrams):
        if dgm is None or len(dgm) == 0:
            out.append((dim, 0, 0.0))
            continue
        dgm = np.asarray(dgm, float)
        finite = np.isfinite(dgm[:, 1])
        dgm = dgm[finite]
        if len(dgm) == 0:
            out.append((dim, 0, 0.0))
            continue
        pers = dgm[:, 1] - dgm[:, 0]
        out.append((dim, int(len(dgm)), float(np.max(pers))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="hepdata_to_dashi output directory")
    ap.add_argument("--out", default="persistence_out", help="output directory")
    ap.add_argument("--maxdim", type=int, default=2, help="max homology dimension for ripser")
    ap.add_argument("--n", type=int, default=0, help="optional subsample size (0=all)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, labels = load_ternary_cloud(inp)
    if args.n and args.n < len(X):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=args.n, replace=False)
        X = X[idx]
        labels = labels[idx]

    try:
        from ripser import ripser
        res = ripser(X, maxdim=args.maxdim, distance_matrix=False)
        dgms = res["dgms"]
        np.savez(out / "ripser_diagrams.npz", *dgms)
        plot_diagrams(dgms, out / "persistence_diagrams.png", "Ternary lens cloud persistence (ripser)")
        summ = summarize(dgms)
        (out / "summary.txt").write_text(
            "\n".join([f"H{dim}: bars={nbar}  max_persistence={maxp}" for dim, nbar, maxp in summ]) + "\n",
            encoding="utf-8",
        )
        print(f"Computed persistence with ripser. Wrote: {out.resolve()}")
        return
    except Exception as e:
        dgms = mst_h0_barcode(X)
        np.savez(out / "mst_h0_diagram.npz", dgms[0])
        plot_diagrams(dgms, out / "persistence_h0_mst.png", "Fallback H0 barcode (MST approximation)")
        summ = summarize(dgms)
        (out / "summary.txt").write_text(
            "ripser unavailable; wrote MST-based H0 approximation.\n"
            f"Error: {e}\n\n" +
            "\n".join([f"H{dim}: bars={nbar}  max_persistence={maxp}" for dim, nbar, maxp in summ]) + "\n",
            encoding="utf-8",
        )
        (out / "ripser_unavailable.txt").write_text(
            "ripser is not installed or failed to import.\n"
            "To enable full H0/H1/H2 persistence:\n"
            "  pip install ripser persim\n\n"
            f"Import error: {e}\n",
            encoding="utf-8",
        )
        print(f"ripser unavailable; wrote fallback H0. Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
