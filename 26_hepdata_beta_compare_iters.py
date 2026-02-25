#!/usr/bin/env python3
"""
26_hepdata_beta_compare_iters.py

Compare beta vectors at selected iterations across all observables.
Default iters: 9,10,12.

Outputs:
  stdout table
  optional CSV to --out
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BETA_COLS = [f"b{k}" for k in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="hepdata_dashi_native",
                    help="Directory containing *_dashi_native_metrics.csv")
    ap.add_argument("--iters", type=str, default="9,10,12",
                    help="Comma-separated iters to extract (e.g. 0,1,9,10,12)")
    ap.add_argument("--out", type=str, default="",
                    help="Optional output CSV path")
    ap.add_argument("--extra", type=str, default="chi2_dof,odd_even_ratio,R_E_hi,alpha",
                    help="Extra columns to include if present")
    args = ap.parse_args()

    root = Path(args.root)
    iters = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
    extra = [x.strip() for x in args.extra.split(",") if x.strip()]

    files = sorted(root.glob("*_dashi_native_metrics.csv"))
    if not files:
        raise SystemExit(f"No *_dashi_native_metrics.csv found under: {root}")

    rows = []
    for f in files:
        label = f.name.replace("_dashi_native_metrics.csv", "")
        df = pd.read_csv(f)
        if "iter" not in df.columns:
            continue
        df = df.sort_values("iter").reset_index(drop=True)

        for it in iters:
            sub = df[df["iter"] == it]
            if sub.empty:
                continue
            r = {"label": label, "iter": it}
            for c in BETA_COLS:
                r[c] = float(sub.iloc[0][c]) if c in sub.columns else np.nan
            for c in extra:
                r[c] = float(sub.iloc[0][c]) if c in sub.columns else np.nan
            rows.append(r)

        # also include final step
        r = {"label": label, "iter": int(df["iter"].iloc[-1]), "_tag": "final"}
        for c in BETA_COLS:
            r[c] = float(df[c].iloc[-1]) if c in df.columns else np.nan
        for c in extra:
            r[c] = float(df[c].iloc[-1]) if c in df.columns else np.nan
        rows.append(r)

    outdf = pd.DataFrame(rows).sort_values(["label", "iter"])
    if outdf.empty:
        raise SystemExit("No rows extracted. Check that 'iter' exists and requested iters are present.")

    # Pretty print
    show_cols = ["label", "iter"] + BETA_COLS + [c for c in extra if c in outdf.columns]
    print(outdf[show_cols].to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        outdf.to_csv(args.out, index=False)
        print(f"\n[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
