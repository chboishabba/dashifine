#!/usr/bin/env python3
"""
38_signature_stability.py

Signature stability under small symmetric perturbations.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh


def signature(G: np.ndarray, tol: float = 1e-10):
    eigs = eigvalsh(G)
    pos = int(np.sum(eigs > tol))
    neg = int(np.sum(eigs < -tol))
    zero = int(len(eigs) - pos - neg)
    return pos, neg, zero, eigs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", required=True, help="Quadratic matrix (npy)")
    ap.add_argument("--out", default="signature_stability.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    G = np.load(args.G)

    records = []

    pos, neg, zero, _ = signature(G, tol=args.tol)
    records.append({"case": "base", "pos": pos, "neg": neg, "zero": zero})

    for eps in [1e-6, 1e-5, 1e-4]:
        noise = eps * rng.standard_normal(G.shape)
        Gp = G + 0.5 * (noise + noise.T)
        pos, neg, zero, _ = signature(Gp, tol=args.tol)
        records.append({"case": f"perturb_eps_{eps}", "pos": pos, "neg": neg, "zero": zero})

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
