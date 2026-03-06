#!/usr/bin/env python3
"""
48_contraction_rate.py

Compute contraction rate: average pairwise distance across labels vs iteration.
Plots log(mean distance) vs t.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv")
    ap.add_argument("--cols", nargs="+", default=["v_pnorm", "v_dnorm", "v_arrow"])
    ap.add_argument("--out", default="viz_contraction_rate.png")
    ap.add_argument("--csv", default="contraction_rate.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    for c in ["label", "iter"] + args.cols:
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")

    rows = []
    for it, g in df.groupby("iter", sort=True):
        X = g[args.cols].values.astype(float)
        if len(X) < 2:
            continue
        # mean pairwise distance
        # compute full pairwise distances efficiently
        dsum = 0.0
        cnt = 0
        for i in range(len(X)):
            diff = X[i+1:] - X[i]
            d = np.linalg.norm(diff, axis=1)
            dsum += d.sum()
            cnt += len(d)
        mean_d = dsum / max(1, cnt)
        rows.append({"iter": int(it), "mean_distance": mean_d})

    out = pd.DataFrame(rows).sort_values("iter")
    out.to_csv(args.csv, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(out["iter"], np.log(out["mean_distance"] + 1e-12), marker="o")
    plt.xlabel("iteration")
    plt.ylabel("log(mean pairwise distance)")
    plt.title("Contraction Rate (log mean distance)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
