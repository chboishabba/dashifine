#!/usr/bin/env python3
"""
35_arrow_shape_independence.py

Test 1: Arrow/Shape independence
  - compute Δa (arrow delta) and Q(Δs) (shape-only quadratic)
  - report correlations and save scatter plot + CSV

Default setup matches current HEPData closure embedding:
  embedding: hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv
  arrow_col: v_depth
  shape_cols: v_pnorm v_dnorm v_arrow
  mask: -1,1,-1
  pos_scale: 0.2034
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_mask(mask_str: str, n: int) -> np.ndarray:
    parts = [float(x.strip()) for x in mask_str.split(",")]
    if len(parts) != n:
        raise SystemExit(f"mask length {len(parts)} != number of shape cols {n}")
    return np.array(parts, dtype=float)


def compute_deltas(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        for i in range(len(g) - 1):
            r0 = g.iloc[i]
            r1 = g.iloc[i + 1]
            row = {"label": label, "iter_from": int(r0["iter"]), "iter_to": int(r1["iter"])}
            for c in cols:
                row[f"d_{c}"] = float(r1[c] - r0[c])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv")
    ap.add_argument("--arrow-col", default="v_depth")
    ap.add_argument("--shape-cols", nargs="+", default=["v_pnorm", "v_dnorm", "v_arrow"])
    ap.add_argument("--mask", default="-1,1,-1", help="comma-separated mask for shape cols")
    ap.add_argument("--pos-scale", type=float, default=0.2034, help="scale for positive mask entries")
    ap.add_argument("--out-prefix", default="arrow_shape_independence")
    ap.add_argument("--forward-only", action="store_true", help="use only steps with Δarrow >= 0")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    needed = ["label", "iter"] + [args.arrow_col] + args.shape_cols
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")

    dcols = [args.arrow_col] + args.shape_cols
    ddf = compute_deltas(df, dcols)

    mask = parse_mask(args.mask, len(args.shape_cols))
    # apply pos_scale to +1 entries
    weights = mask.copy()
    weights[weights > 0] = args.pos_scale

    d_arrow = ddf[f"d_{args.arrow_col}"].values
    d_shape = np.vstack([ddf[f"d_{c}"].values for c in args.shape_cols]).T

    # Q(Δs) = sum_i w_i * (Δs_i)^2
    q_vals = np.sum(weights * (d_shape ** 2), axis=1)

    if args.forward_only:
        m = d_arrow >= 0
        d_arrow = d_arrow[m]
        q_vals = q_vals[m]
        ddf = ddf[m].reset_index(drop=True)

    # correlations
    if len(d_arrow) < 2:
        raise SystemExit("not enough steps to compute correlations")
    pearson = np.corrcoef(d_arrow, q_vals)[0, 1]
    # Spearman via rank correlation (simple, no scipy)
    da_rank = pd.Series(d_arrow).rank().values
    q_rank = pd.Series(q_vals).rank().values
    spearman = np.corrcoef(da_rank, q_rank)[0, 1]

    print("Arrow/Shape independence")
    print(f"steps: {len(d_arrow)}")
    print(f"pearson_r: {pearson:.6g}")
    print(f"spearman_r: {spearman:.6g}")

    # save CSV
    out_csv = f"{args.out_prefix}.csv"
    out_png = f"{args.out_prefix}.png"
    out_txt = f"{args.out_prefix}_summary.txt"

    out_df = ddf.copy()
    out_df["delta_arrow"] = d_arrow
    out_df["Q_shape"] = q_vals
    out_df.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"steps: {len(d_arrow)}\n")
        f.write(f"pearson_r: {pearson}\n")
        f.write(f"spearman_r: {spearman}\n")

    # scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(d_arrow, q_vals, s=8, alpha=0.5)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(f"Δ{args.arrow_col}")
    plt.ylabel("Q(Δshape)")
    title = "Arrow vs Q(Δshape)"
    if args.forward_only:
        title += " (forward only)"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(f"Wrote {out_csv}, {out_png}, {out_txt}")


if __name__ == "__main__":
    main()
