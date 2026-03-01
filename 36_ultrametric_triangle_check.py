#!/usr/bin/env python3
"""
36_ultrametric_triangle_check.py

Test 4: Strong ultrametric triangle inequality on ternary lens trajectories.
Checks d(x,z) <= max(d(x,y), d(y,z)) for random triples within each label.

Distance:
  - agreement count over ternary coordinates (L0..L9)
  - d = 2^{-agree}
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd


def load_ternary(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("L")]
    if not cols:
        raise SystemExit(f"No L* columns in {path}")
    X = df[cols].astype(float).values
    return X


def lcp_depth(a: np.ndarray, b: np.ndarray) -> int:
    """
    Longest common prefix depth (first-difference index).
    Assumes coordinates are ordered coarse->fine.
    """
    diff = a != b
    if not np.any(diff):
        return int(a.shape[0])
    return int(np.argmax(diff))


def dist_ultra(a: np.ndarray, b: np.ndarray, base: float = 2.0) -> float:
    depth = lcp_depth(a, b)
    return base ** (-float(depth))


def check_label(X: np.ndarray, trials: int, rng: random.Random, base: float) -> tuple[int, float]:
    n = len(X)
    if n < 3:
        return 0, 0.0
    violations = 0
    max_violation = 0.0
    for _ in range(trials):
        i, j, k = rng.sample(range(n), 3)
        x, y, z = X[i], X[j], X[k]
        lhs = dist_ultra(x, z, base=base)
        rhs = max(dist_ultra(x, y, base=base), dist_ultra(y, z, base=base))
        if lhs > rhs:
            violations += 1
            max_violation = max(max_violation, lhs - rhs)
    return violations, max_violation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="hepdata_to_dashi_all", help="root containing <label>/lenses_ternary.csv")
    ap.add_argument("--trials", type=int, default=5000, help="random triples per label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=float, default=2.0, help="ultrametric base (distance = base^{-depth})")
    ap.add_argument("--out", default="ultrametric_triangle_report.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    rows = []
    total_violations = 0
    total_trials = 0
    global_max_violation = 0.0

    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        f = d / "lenses_ternary.csv"
        if not f.exists():
            continue
        X = load_ternary(f)
        v, m = check_label(X, args.trials, rng, base=args.base)
        rows.append({
            "label": d.name,
            "n_points": len(X),
            "trials": args.trials,
            "violations": v,
            "max_violation": m,
        })
        total_violations += v
        total_trials += args.trials
        global_max_violation = max(global_max_violation, m)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("Ultrametric triangle check")
    print(f"labels: {len(rows)}")
    print(f"total_trials: {total_trials}")
    print(f"total_violations: {total_violations}")
    print(f"global_max_violation: {global_max_violation}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
