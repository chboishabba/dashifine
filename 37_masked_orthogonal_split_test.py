#!/usr/bin/env python3
"""
37_masked_orthogonal_split_test.py

Empirical tests for Masked Orthogonal Split lemma:
  1) G-self-adjointness of P: ||P^T G - G P||_F
  2) Cross-term vanishing: <Δs_P, Δs_⊥>_G ≈ 0
  3) Quadratic split: Q(Δs) ≈ Q(Δs_P) + Q(Δs_⊥)

Inputs:
  - Δs (shape deltas): provided as .npy or computed from embedding CSV
  - G (shape-only quadratic form): provided as .npy or built from mask+pos_scale
  - P (projection): provided as .npy

Outputs:
  - masked_orthogonal_split_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_mask(mask_str: str, n: int) -> np.ndarray:
    parts = [float(x.strip()) for x in mask_str.split(",")]
    if len(parts) != n:
        raise SystemExit(f"mask length {len(parts)} != number of shape cols {n}")
    return np.array(parts, dtype=float)


def build_G_from_mask(mask: np.ndarray, pos_scale: float) -> np.ndarray:
    w = mask.copy()
    w[w > 0] = pos_scale
    return np.diag(w)


def compute_deltas_from_embedding(path: Path, shape_cols: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    needed = ["label", "iter"] + shape_cols
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")
    rows = []
    for _, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        for i in range(len(g) - 1):
            r0 = g.iloc[i]
            r1 = g.iloc[i + 1]
            rows.append([float(r1[c] - r0[c]) for c in shape_cols])
    if not rows:
        raise SystemExit("no deltas computed from embedding")
    return np.array(rows, dtype=float)


def inner_G(u: np.ndarray, v: np.ndarray, G: np.ndarray) -> float:
    return float(u @ G @ v)


def Q(v: np.ndarray, G: np.ndarray) -> float:
    return inner_G(v, v, G)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-shape", help="path to Δs .npy (N,d)")
    ap.add_argument("--embedding", help="closure embedding CSV for computing Δs")
    ap.add_argument("--shape-cols", nargs="+", default=["v_pnorm", "v_dnorm", "v_arrow"])
    ap.add_argument("--G", help="path to G matrix .npy (d,d)")
    ap.add_argument("--mask", help="comma-separated mask for diagonal G")
    ap.add_argument("--pos-scale", type=float, default=0.2034)
    ap.add_argument("--P", required=True, help="path to projection matrix P .npy (d,d)")
    ap.add_argument("--out", default="masked_orthogonal_split_report.csv")
    args = ap.parse_args()

    # Δs
    if args.delta_shape:
        dS = np.load(args.delta_shape)
    elif args.embedding:
        dS = compute_deltas_from_embedding(Path(args.embedding), args.shape_cols)
    else:
        raise SystemExit("provide --delta-shape or --embedding")

    if dS.ndim != 2:
        raise SystemExit("Δs must be 2D (N,d)")
    N, d = dS.shape

    # G
    if args.G:
        G = np.load(args.G)
    elif args.mask:
        mask = parse_mask(args.mask, d)
        G = build_G_from_mask(mask, args.pos_scale)
    else:
        raise SystemExit("provide --G or --mask")

    if G.shape != (d, d):
        raise SystemExit(f"G shape {G.shape} incompatible with d={d}")

    # P
    P = np.load(args.P)
    if P.shape != (d, d):
        raise SystemExit(f"P shape {P.shape} incompatible with d={d}")

    # 1) G-self-adjointness
    self_adj_error = float(np.linalg.norm(P.T @ G - G @ P, ord="fro"))

    # 2) Cross-terms + 3) split residuals
    cross_terms = np.zeros(N, dtype=float)
    split_resid = np.zeros(N, dtype=float)
    for i in range(N):
        v = dS[i]
        vP = P @ v
        v_perp = v - vP
        cross_terms[i] = inner_G(vP, v_perp, G)
        split_resid[i] = Q(v, G) - (Q(vP, G) + Q(v_perp, G))

    report = {
        "self_adj_error": self_adj_error,
        "cross_mean": float(np.mean(cross_terms)),
        "cross_std": float(np.std(cross_terms)),
        "cross_max_abs": float(np.max(np.abs(cross_terms))),
        "energy_residual_mean": float(np.mean(split_resid)),
        "energy_residual_std": float(np.std(split_resid)),
        "energy_residual_max_abs": float(np.max(np.abs(split_resid))),
        "n_steps": int(N),
        "dim": int(d),
    }

    pd.DataFrame([report]).to_csv(args.out, index=False)
    print("Masked orthogonal split test")
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
