#!/usr/bin/env python3
"""
Delta-cone signature screen (indefinite-only option).

Key ideas:
- Choose an "arrow" coordinate used ONLY for monotonicity / forward filtering.
- Choose x-cols used for Q(Δx) test (arrow col excluded by default).
- Score cone on forward steps: cone_ok among steps with arrow_ok.
- Optionally require nondegenerate (z=0) and/or indefinite (p>0 and q>0).

Outputs: delta_cone_signature_rank.csv
"""
from __future__ import annotations
import argparse
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

def parse_cols(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]

def iter_signatures(d: int, allow_zero: bool) -> List[Tuple[int,int,int,np.ndarray]]:
    # Enumerate all sign masks in {-1, +1}^d (or include 0 if allow_zero)
    vals = [-1, 1] if not allow_zero else [-1, 0, 1]
    out = []
    for mask in itertools.product(vals, repeat=d):
        mask = np.array(mask, dtype=float)
        p = int(np.sum(mask > 0))
        q = int(np.sum(mask < 0))
        z = int(np.sum(mask == 0))
        out.append((p,q,z,mask))
    # stable ordering: prefer fewer zeros, then more balanced, then lex
    out.sort(key=lambda t: (t[2], abs(t[0]-t[1]), -t[0], -t[1]))
    return out

def quad_form(mask: np.ndarray, dX: np.ndarray) -> np.ndarray:
    # mask shape (d,), dX shape (n,d)
    return (dX * dX * mask.reshape(1,-1)).sum(axis=1)

@dataclass
class Score:
    p: int
    q: int
    z: int
    mask: str
    forward_frac_min: float
    forward_frac_mean: float
    cone_forward_frac_min: float
    cone_forward_frac_mean: float
    max_Qd_violation_max: float
    n_labels: int
    n_steps_total: int

def score_signature(df: pd.DataFrame, x_cols: List[str], arrow_col: str,
                    mask: np.ndarray, eps_q: float, eps_arrow: float) -> Tuple[Score, Dict]:
    # Requires df has columns: label, step, x_cols, arrow_col
    labels = sorted(df["label"].unique().tolist())
    forward_fracs = []
    cone_fracs = []
    max_viol = []
    total_steps = 0

    for lab in labels:
        g = df[df["label"]==lab].sort_values("step")
        X = g[x_cols].to_numpy(dtype=float)
        A = g[arrow_col].to_numpy(dtype=float)

        if len(g) < 2:
            continue
        dX = X[1:] - X[:-1]
        dA = A[1:] - A[:-1]

        forward = dA >= (-eps_arrow)
        qd = quad_form(mask, dX)

        cone_ok = qd <= eps_q

        n = len(qd)
        total_steps += n

        if n == 0:
            continue

        ffrac = float(np.mean(forward)) if n else 0.0
        forward_fracs.append(ffrac)

        if np.any(forward):
            cfrac = float(np.mean(cone_ok[forward]))
            # violation among forward steps
            viol = float(np.max(qd[forward] - eps_q))
        else:
            cfrac = 0.0
            viol = float(np.max(qd - eps_q))
        cone_fracs.append(cfrac)
        max_viol.append(viol)

    # Aggregate
    if not labels:
        raise RuntimeError("No labels found. Expect a 'label' column.")

    def safe_mean(xs): return float(np.mean(xs)) if xs else 0.0
    def safe_min(xs): return float(np.min(xs)) if xs else 0.0
    def safe_max(xs): return float(np.max(xs)) if xs else 0.0

    score = Score(
        p=int(np.sum(mask>0)), q=int(np.sum(mask<0)), z=int(np.sum(mask==0)),
        mask=",".join(str(int(m)) for m in mask),
        forward_frac_min=safe_min(forward_fracs),
        forward_frac_mean=safe_mean(forward_fracs),
        cone_forward_frac_min=safe_min(cone_fracs),
        cone_forward_frac_mean=safe_mean(cone_fracs),
        max_Qd_violation_max=safe_max(max_viol),
        n_labels=len(labels),
        n_steps_total=int(total_steps),
    )
    details = {"labels": labels}
    return score, details

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True, help="CSV path (must include columns for label, step, and embedding coords)")
    ap.add_argument("--arrow-col", required=True, help="Column used for forward filtering / monotonicity")
    ap.add_argument("--x-cols", default=None, help="Comma-separated list of columns to use for Q (default: all v_* except label/step and arrow-col)")
    ap.add_argument("--allow-zero", action="store_true", help="Allow degenerate masks (0 entries)")
    ap.add_argument("--require-nondegenerate", action="store_true", help="Require z=0")
    ap.add_argument("--require-indefinite", action="store_true", help="Require p>0 and q>0")
    ap.add_argument("--eps", type=float, default=1e-12, help="Cone tolerance for Q(dX) <= eps")
    ap.add_argument("--eps-arrow", type=float, default=1e-12, help="Arrow tolerance for dA >= -eps_arrow")
    ap.add_argument("--min-cone-frac", type=float, default=0.95, help="Threshold on cone_forward_frac_min")
    ap.add_argument("--min-forward-frac", type=float, default=0.5, help="Require forward_frac_min >= this (0 disables)")
    ap.add_argument("--out", default="delta_cone_signature_rank.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    # normalize expected columns
    if "label" not in df.columns:
        # try common alternatives
        for cand in ["key", "series", "run", "traj", "group"]:
            if cand in df.columns:
                df = df.rename(columns={cand:"label"})
                break
    if "step" not in df.columns:
        for cand in ["t", "time", "iter", "k", "index"]:
            if cand in df.columns:
                df = df.rename(columns={cand:"step"})
                break
    if "label" not in df.columns or "step" not in df.columns:
        raise SystemExit("[err] embedding csv must contain 'label' and 'step' columns (or recognizable aliases).")

    if args.arrow_col not in df.columns:
        raise SystemExit(f"[err] arrow col '{args.arrow_col}' not found in csv.")

    x_cols = parse_cols(args.x_cols)
    if x_cols is None:
        # default heuristic: use v_* numeric columns, excluding label/step and arrow
        vcols = [c for c in df.columns if c.startswith("v_")]
        x_cols = [c for c in vcols if c not in ["label","step", args.arrow_col]]
        # also exclude obviously non-numeric
        for c in ["v_label","v_step"]:
            if c in x_cols: x_cols.remove(c)

    # ensure no arrow in x
    x_cols = [c for c in x_cols if c != args.arrow_col]

    # validate numeric
    for c in x_cols + [args.arrow_col]:
        if c not in df.columns:
            raise SystemExit(f"[err] column '{c}' not in csv.")
    d = len(x_cols)
    if d == 0:
        raise SystemExit("[err] no x-cols selected for Q test.")
    print(f"[info] x cols used: {x_cols}")
    print(f"[info] arrow col: {args.arrow_col}")

    sigs = iter_signatures(d, allow_zero=args.allow_zero)
    rows = []
    best = None

    for p,q,z,mask in sigs:
        if args.require_nondegenerate and z != 0:
            continue
        if args.require_indefinite and not (p > 0 and q > 0):
            continue

        sc, _ = score_signature(df, x_cols, args.arrow_col, mask, eps_q=args.eps, eps_arrow=args.eps_arrow)

        # thresholds
        if args.min_forward_frac > 0 and sc.forward_frac_min < args.min_forward_frac:
            passed = False
        else:
            passed = sc.cone_forward_frac_min >= args.min_cone_frac

        rows.append({
            "p": sc.p, "q": sc.q, "z": sc.z, "mask": sc.mask,
            "forward_frac_min": sc.forward_frac_min,
            "forward_frac_mean": sc.forward_frac_mean,
            "cone_forward_frac_min": sc.cone_forward_frac_min,
            "cone_forward_frac_mean": sc.cone_forward_frac_mean,
            "max_Qd_violation_max": sc.max_Qd_violation_max,
            "passed": bool(passed),
        })

        if passed:
            if best is None:
                best = sc
            else:
                # ranking: primary cone_forward_frac_min, then fewer zeros, then higher forward_frac_min, then smaller violation
                cur = (sc.cone_forward_frac_min, -sc.z, sc.forward_frac_min, -sc.max_Qd_violation_max)
                b   = (best.cone_forward_frac_min, -best.z, best.forward_frac_min, -best.max_Qd_violation_max)
                if cur > b:
                    best = sc

    out = pd.DataFrame(rows)
    # sort for inspection
    out = out.sort_values(
        ["passed","cone_forward_frac_min","forward_frac_min","cone_forward_frac_mean","max_Qd_violation_max","z"],
        ascending=[False,False,False,False,True,True]
    )
    out.to_csv(args.out, index=False)

    if best is None:
        print("[warn] no signature passed thresholds.")
        print("[hint] try lowering --min-cone-frac and/or --min-forward-frac, or add --allow-zero to inspect degenerates.")
    else:
        print("=== Best delta-cone signature under constraints ===")
        for k,v in best.__dict__.items():
            print(f"{k:25s} {v}")
    print(f"[ok] wrote: {args.out}")
    print("[note] Scores are cone fractions CONDITIONAL on forward steps (per arrow).")
    print("[note] Arrow is NOT included in Q; it is only used for forward filtering.")

if __name__ == "__main__":
    main()
