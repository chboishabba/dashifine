#!/usr/bin/env python3
"""
Strict defect monotonicity test for DASHI LHC runs.

Tests:
    ||D(x_{t+1})||_1 <= ||D(x_t)||_1
Optionally:
    L2 defect
    weighted-MDL defect

Outputs:
    defect_monotonicity_report.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def parse_int_list(s):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def load_timeseries(path, dim):
    df = pd.read_csv(path)
    cols = ["label", "iter"] + [f"b{i}" for i in range(dim)]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    return df.sort_values(["label", "iter"]).reset_index(drop=True)


def compute_defect_norms(betas, defect_idx, w_defect=1.0):
    """
    D(x) = x - P(x) = coordinates in defect_idx
    """
    D = betas[:, list(defect_idx)]
    L1 = np.sum(np.abs(D), axis=1)
    L2 = np.sqrt(np.sum(D * D, axis=1))
    MDL = w_defect * L1
    return L1, L2, MDL


def test_defect_monotonicity(df, dim, defect_idx, eps=0.0, w_defect=1.0):
    rows = []

    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        betas = g[[f"b{i}" for i in range(dim)]].to_numpy(dtype=float)
        iters = g["iter"].to_numpy()

        L1, L2, MDL = compute_defect_norms(betas, defect_idx, w_defect=w_defect)

        def check_monotone(arr):
            diffs = arr[1:] - arr[:-1]
            violations = np.where(diffs > eps)[0]
            return (
                len(violations) == 0,
                len(violations),
                float(diffs.max()) if len(diffs) else 0.0,
                int(iters[violations[0]+1]) if len(violations) else -1,
            )

        L1_ok, L1_n, L1_worst, L1_first = check_monotone(L1)
        L2_ok, L2_n, L2_worst, L2_first = check_monotone(L2)
        MDL_ok, MDL_n, MDL_worst, MDL_first = check_monotone(MDL)

        rows.append({
            "label": label,
            "L1_monotone": L1_ok,
            "L1_violations": L1_n,
            "L1_worst_increase": L1_worst,
            "L1_first_violation_iter": L1_first,
            "L2_monotone": L2_ok,
            "L2_violations": L2_n,
            "L2_worst_increase": L2_worst,
            "L2_first_violation_iter": L2_first,
            "MDL_defect_monotone": MDL_ok,
            "MDL_defect_violations": MDL_n,
            "MDL_defect_worst_increase": MDL_worst,
            "MDL_defect_first_violation_iter": MDL_first,
            "last_iter": int(iters.max()),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=5)
    ap.add_argument("--defect-idx", default="1,3")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--w-defect", type=float, default=1.0)
    args = ap.parse_args()

    ensure_dir(args.out)

    defect_idx = parse_int_list(args.defect_idx)
    df = load_timeseries(args.timeseries, args.dim)

    report = test_defect_monotonicity(
        df,
        dim=args.dim,
        defect_idx=defect_idx,
        eps=args.eps,
        w_defect=args.w_defect,
    )

    out_path = os.path.join(args.out, "defect_monotonicity_report.csv")
    report.to_csv(out_path, index=False)
    print("[ok] wrote:", out_path)


if __name__ == "__main__":
    main()
