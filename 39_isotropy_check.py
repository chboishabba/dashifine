#!/usr/bin/env python3
"""
39_isotropy_check.py

Isotropy compatibility test in the shape subspace (coords [0,1]).
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def rotation(theta: float, dim: int) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s, c]])
    R = np.eye(dim)
    R[:2, :2] = R2
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", required=True, help="Quadratic matrix (npy)")
    ap.add_argument("--out", default="isotropy_check.csv")
    ap.add_argument("--steps", type=int, default=50, help="number of rotation samples")
    args = ap.parse_args()

    G = np.load(args.G)
    dim = G.shape[0]

    records = []
    for theta in np.linspace(0.0, 2.0 * np.pi, args.steps, endpoint=True):
        R = rotation(theta, dim)
        G_rot = R.T @ G @ R
        diff = float(np.linalg.norm(G_rot - G, ord="fro"))
        records.append({"theta": float(theta), "fro_error": diff})

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print("Max isotropy deviation:", df["fro_error"].max())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
