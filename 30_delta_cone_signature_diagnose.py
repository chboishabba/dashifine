#!/usr/bin/env python3
"""
Delta-cone signature screen + diagnostics.

Key ideas:
- Choose an "arrow" coordinate (monotone filter) but DO NOT include it in Q.
- Test cone on deltas of x-cols only: Q(dX) <= eps_cone.
- Score cone fraction CONDITIONAL on forward steps (per arrow).
- Optionally require nondegenerate (z=0) and/or indefinite (p>0 and q>0).
- Diagnostics: dump worst violating forward steps for a chosen mask (or best mask).
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True, help="CSV with per-step embedding rows.")
    ap.add_argument("--label-col", default="label", help="Label/group column.")
    ap.add_argument("--step-col", default="step", help="Step index column.")
    ap.add_argument("--arrow-col", required=True, help="Arrow column used for forward filtering.")
    ap.add_argument("--x-cols", default="", help="Comma-separated x cols. If empty, auto-select numeric cols excluding label/step/arrow.")
    ap.add_argument("--eps", type=float, default=1e-12, help="Cone tolerance: require Q(dX) <= eps.")
    ap.add_argument("--eps-arrow", type=float, default=1e-12, help="Arrow tolerance: forward if dA >= -eps_arrow.")
    ap.add_argument("--require-nondegenerate", action="store_true", help="Require z=0.")
    ap.add_argument("--require-indefinite", action="store_true", help="Require p>0 and q>0.")
    ap.add_argument("--allow-zero", action="store_true", help="Allow degenerate signatures; otherwise exclude z>0.")
    ap.add_argument("--min-forward-frac", type=float, default=0.0)
    ap.add_argument("--min-cone-frac", type=float, default=0.0)
    ap.add_argument("--out-rank", default="delta_cone_signature_rank.csv")
    # optional axis/weight handling
    ap.add_argument("--auto-null-axis", action="store_true",
                    help="Drop x-cols whose delta-std is near-zero (treat as null axis).")
    ap.add_argument("--null-std-abs", type=float, default=1e-9,
                    help="Absolute std threshold for null-axis detection (on deltas).")
    ap.add_argument("--null-std-rel", type=float, default=1e-3,
                    help="Relative std threshold vs median delta-std for null-axis detection.")
    ap.add_argument("--diag-weights", default="",
                    help="Optional comma-separated diagonal weights (len = #x-cols).")
    ap.add_argument("--fit-posneg-scale", action="store_true",
                    help="Fit a single scale on +1 axes vs -1 axes to reduce violations.")
    ap.add_argument("--posneg-cap", type=float, default=1.0,
                    help="Upper cap on fitted pos-scale (default 1.0 = only shrink + axes).")
    # diagnostics
    ap.add_argument("--mask", default="", help="Mask like '1,-1,-1'. If omitted, uses best under constraints.")
    ap.add_argument("--dump-violations", action="store_true", help="Write a CSV of forward-step cone violations for chosen/best mask.")
    ap.add_argument("--violations-csv", default="delta_cone_violations.csv")
    ap.add_argument("--top-k", type=int, default=50, help="Keep top-K largest Q(dX) violations.")
    return ap.parse_args()


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def select_x_cols(df: pd.DataFrame, label_col: str, step_col: str, arrow_col: str, x_cols_arg: str) -> List[str]:
    if x_cols_arg.strip():
        cols = [c.strip() for c in x_cols_arg.split(",") if c.strip()]
        return cols
    exclude = {label_col, step_col, arrow_col}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if is_numeric_series(df[c]):
            cols.append(c)
    return cols


def all_masks(d: int, allow_zero: bool) -> List[Tuple[int,int,int,str,np.ndarray]]:
    """
    Generate all sign masks in {+1,-1}^d (and optionally 0 for degenerate).
    Returns list of (p,q,z,mask_str,mask_vec).
    """
    vals = [1, -1] if not allow_zero else [1, -1, 0]
    masks = []
    def rec(i, cur):
        if i == d:
            v = np.array(cur, dtype=float)
            p = int(np.sum(v == 1))
            q = int(np.sum(v == -1))
            z = int(np.sum(v == 0))
            mstr = ",".join(str(int(x)) for x in v)
            masks.append((p,q,z,mstr,v))
            return
        for x in vals:
            rec(i+1, cur+[x])
    rec(0, [])
    # stable order: prefer fewer zeros, then higher p+q (nondeg), then whatever
    masks.sort(key=lambda t: (t[2], -(t[0]+t[1]), -t[0], t[1], t[3]))
    return masks


def q_of_dx(dx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # dx: (n, d), mask: (d,)
    return (dx * dx) @ mask


def q_of_dx_weighted(dx: np.ndarray, mask: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None:
        return q_of_dx(dx, mask)
    return (dx * dx) @ (mask * weights)


def compute_delta_stds(df: pd.DataFrame, label_col: str, step_col: str, x_cols: List[str]) -> Dict[str, float]:
    deltas = []
    for _, g in df.groupby(label_col, sort=False):
        g = g.sort_values(step_col)
        X = g[x_cols].to_numpy(dtype=float)
        if len(X) < 2:
            continue
        deltas.append(X[1:] - X[:-1])
    if not deltas:
        return {c: 0.0 for c in x_cols}
    dX = np.vstack(deltas)
    stds = np.std(dX, axis=0)
    return {c: float(s) for c, s in zip(x_cols, stds)}


@dataclass
class Score:
    p: int
    q: int
    z: int
    mask: str
    pos_scale: float
    forward_frac_min: float
    forward_frac_mean: float
    cone_forward_frac_min: float
    cone_forward_frac_mean: float
    max_Qd_violation_max: float
    passed: bool


def score_mask(df: pd.DataFrame, label_col: str, step_col: str, x_cols: List[str], arrow_col: str,
              mask_vec: np.ndarray, eps_cone: float, eps_arrow: float,
              weights: Optional[np.ndarray], fit_posneg_scale: bool, posneg_cap: float,
              min_forward_frac: float, min_cone_frac: float) -> Score:
    pos_scale = 1.0
    if fit_posneg_scale:
        pos_idx = mask_vec == 1
        neg_idx = mask_vec == -1
        max_scales = []
        for _, g in df.groupby(label_col, sort=False):
            g = g.sort_values(step_col)
            X = g[x_cols].to_numpy(dtype=float)
            A = g[arrow_col].to_numpy(dtype=float)
            if len(X) < 2:
                continue
            dX = X[1:] - X[:-1]
            dA = A[1:] - A[:-1]
            forward = dA >= -eps_arrow
            if not np.any(forward):
                continue
            dXf = dX[forward]
            if weights is not None:
                w = weights
                pos_sum = np.sum((dXf[:, pos_idx] ** 2) * w[pos_idx], axis=1) if np.any(pos_idx) else None
                neg_sum = np.sum((dXf[:, neg_idx] ** 2) * w[neg_idx], axis=1) if np.any(neg_idx) else None
            else:
                pos_sum = np.sum(dXf[:, pos_idx] ** 2, axis=1) if np.any(pos_idx) else None
                neg_sum = np.sum(dXf[:, neg_idx] ** 2, axis=1) if np.any(neg_idx) else None
            if pos_sum is None or neg_sum is None:
                continue
            valid = pos_sum > 0
            if not np.any(valid):
                continue
            ratios = (neg_sum[valid] + eps_cone) / pos_sum[valid]
            scales = np.sqrt(np.clip(ratios, a_min=0.0, a_max=None))
            if len(scales):
                max_scales.append(float(np.min(scales)))
        if len(max_scales):
            pos_scale = min(float(np.min(max_scales)), float(posneg_cap))

    eff_weights = weights
    if pos_scale != 1.0:
        w = np.ones(len(mask_vec), dtype=float) if weights is None else np.array(weights, dtype=float)
        w = w * 1.0
        w[mask_vec == 1] *= pos_scale ** 2
        eff_weights = w

    # compute per-label stats
    per_label_forward = []
    per_label_cone_forward = []
    per_label_max_violation = []

    for lab, g in df.groupby(label_col, sort=False):
        g = g.sort_values(step_col)
        X = g[x_cols].to_numpy(dtype=float)
        A = g[arrow_col].to_numpy(dtype=float)

        if len(X) < 2:
            continue

        dX = X[1:] - X[:-1]
        dA = A[1:] - A[:-1]

        forward = dA >= -eps_arrow
        forward_frac = float(np.mean(forward)) if len(forward) else 0.0

        qd = q_of_dx_weighted(dX, mask_vec, eff_weights)
        cone_ok = qd <= eps_cone

        if np.any(forward):
            cone_forward_frac = float(np.mean(cone_ok[forward]))
            max_violation = float(np.max(qd[forward] - eps_cone))
        else:
            # no forward steps: treat as 0 forward_frac and 0 cone_forward_frac
            cone_forward_frac = 0.0
            max_violation = float("inf")

        per_label_forward.append(forward_frac)
        per_label_cone_forward.append(cone_forward_frac)
        per_label_max_violation.append(max_violation)

    if not per_label_forward:
        return Score(
            p=int(np.sum(mask_vec==1)), q=int(np.sum(mask_vec==-1)), z=int(np.sum(mask_vec==0)),
            mask=",".join(str(int(x)) for x in mask_vec),
            pos_scale=pos_scale,
            forward_frac_min=0.0, forward_frac_mean=0.0,
            cone_forward_frac_min=0.0, cone_forward_frac_mean=0.0,
            max_Qd_violation_max=float("inf"),
            passed=False
        )

    fmin = float(np.min(per_label_forward))
    fmean = float(np.mean(per_label_forward))
    cmin = float(np.min(per_label_cone_forward))
    cmean = float(np.mean(per_label_cone_forward))
    vmax = float(np.max(per_label_max_violation))

    passed = (fmin >= min_forward_frac) and (cmin >= min_cone_frac)

    return Score(
        p=int(np.sum(mask_vec==1)), q=int(np.sum(mask_vec==-1)), z=int(np.sum(mask_vec==0)),
        mask=",".join(str(int(x)) for x in mask_vec),
        pos_scale=pos_scale,
        forward_frac_min=fmin, forward_frac_mean=fmean,
        cone_forward_frac_min=cmin, cone_forward_frac_mean=cmean,
        max_Qd_violation_max=vmax if math.isfinite(vmax) else vmax,
        passed=passed
    )


def dump_violations(df: pd.DataFrame, label_col: str, step_col: str, x_cols: List[str], arrow_col: str,
                    mask_vec: np.ndarray, eps_cone: float, eps_arrow: float,
                    weights: Optional[np.ndarray], pos_scale: float,
                    out_csv: str, top_k: int):
    eff_weights = weights
    if pos_scale != 1.0:
        w = np.ones(len(mask_vec), dtype=float) if weights is None else np.array(weights, dtype=float)
        w = w * 1.0
        w[mask_vec == 1] *= pos_scale ** 2
        eff_weights = w

    rows = []
    for lab, g in df.groupby(label_col, sort=False):
        g = g.sort_values(step_col)
        X = g[x_cols].to_numpy(dtype=float)
        A = g[arrow_col].to_numpy(dtype=float)
        steps = g[step_col].to_numpy()

        if len(X) < 2:
            continue

        dX = X[1:] - X[:-1]
        dA = A[1:] - A[:-1]
        forward = dA >= -eps_arrow

        qd = q_of_dx_weighted(dX, mask_vec, eff_weights)
        viol = forward & (qd > eps_cone)

        idxs = np.where(viol)[0]
        for i in idxs:
            r = {
                label_col: lab,
                "step_t": int(steps[i]),
                "step_t1": int(steps[i+1]),
                "A_t": float(A[i]),
                "A_t1": float(A[i+1]),
                "dA": float(dA[i]),
                "Qd": float(qd[i]),
                "Qd_minus_eps": float(qd[i] - eps_cone),
            }
            for j, c in enumerate(x_cols):
                r[f"d{c}"] = float(dX[i, j])
            rows.append(r)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("Qd_minus_eps", ascending=False).head(top_k)
    out.to_csv(out_csv, index=False)


def main():
    args = parse_args()
    df = pd.read_csv(args.embedding)

    if args.step_col not in df.columns and args.step_col == "step" and "iter" in df.columns:
        print("[warn] missing 'step' column; falling back to 'iter'.")
        args.step_col = "iter"

    # basic column checks
    for c in [args.label_col, args.step_col, args.arrow_col]:
        if c not in df.columns:
            raise SystemExit(f"[err] missing required column: {c}")

    x_cols = select_x_cols(df, args.label_col, args.step_col, args.arrow_col, args.x_cols)
    if len(x_cols) == 0:
        raise SystemExit("[err] no x cols selected")
    d = len(x_cols)
    orig_x_cols = list(x_cols)

    weights = None
    if args.diag_weights.strip():
        weights = np.array([float(x) for x in args.diag_weights.split(",")], dtype=float)
        if len(weights) != d:
            raise SystemExit(f"[err] diag-weights len={len(weights)} != #x-cols={d}")

    if args.auto_null_axis:
        stds = compute_delta_stds(df, args.label_col, args.step_col, x_cols)
        std_vals = np.array([stds[c] for c in x_cols], dtype=float)
        median_std = float(np.median(std_vals)) if len(std_vals) else 0.0
        thresh = max(args.null_std_abs, args.null_std_rel * median_std)
        keep_cols = [c for c in x_cols if stds[c] > thresh]
        dropped = [c for c in x_cols if c not in keep_cols]
        if dropped:
            print(f"[warn] dropping near-null axes (delta-std <= {thresh:g}): {dropped}")
            x_cols = keep_cols
            d = len(x_cols)
            if weights is not None:
                keep_idx = [i for i, c in enumerate(orig_x_cols) if c in x_cols]
                weights = np.array([weights[i] for i in keep_idx], dtype=float)
        if d == 0:
            raise SystemExit("[err] all x cols dropped as null; relax thresholds or disable auto-null-axis")

    print(f"[info] x cols used: {x_cols}")
    print(f"[info] arrow col: {args.arrow_col}")
    if weights is not None:
        print(f"[info] diag weights: {weights.tolist()}")
    if args.fit_posneg_scale:
        print(f"[info] fitting pos/neg scale with cap={args.posneg_cap}")

    masks = all_masks(d, allow_zero=args.allow_zero)

    scores: List[Score] = []
    for p,q,z,mstr,mvec in masks:
        if not args.allow_zero and z > 0:
            continue
        if args.require_nondegenerate and z != 0:
            continue
        if args.require_indefinite and not (p > 0 and q > 0):
            continue
        sc = score_mask(
            df, args.label_col, args.step_col, x_cols, args.arrow_col,
            mvec, args.eps, args.eps_arrow,
            weights, args.fit_posneg_scale, args.posneg_cap,
            args.min_forward_frac, args.min_cone_frac
        )
        scores.append(sc)

    out_rank = pd.DataFrame([s.__dict__ for s in scores])
    # rank: cone_forward_frac_min (desc), then max_violation (asc), then forward_frac_min (desc), then fewer zeros
    out_rank = out_rank.sort_values(
        ["cone_forward_frac_min", "max_Qd_violation_max", "forward_frac_min", "z", "p", "q"],
        ascending=[False, True, False, True, False, False],
    )
    out_rank.to_csv(args.out_rank, index=False)
    print(f"[ok] wrote: {args.out_rank}")
    print("[note] Scores are cone fractions CONDITIONAL on forward steps (per arrow).")
    print("[note] Arrow is NOT included in Q; it is only used for forward filtering.")

    # choose mask for diagnostics / printing
    chosen = None
    if args.mask.strip():
        m = args.mask.strip().replace(" ", "")
        chosen = out_rank[out_rank["mask"] == m].head(1)
        if len(chosen) == 0:
            raise SystemExit(f"[err] mask not found in rank table: {m}")
    else:
        chosen = out_rank.head(1)

    if len(chosen):
        print("=== Best/Chosen delta-cone signature under constraints ===")
        print(chosen.iloc[0].to_string())

        if args.dump_violations:
            mask_str = chosen.iloc[0]["mask"]
            mask_vec = np.array([float(x) for x in mask_str.split(",")], dtype=float)
            pos_scale = float(chosen.iloc[0].get("pos_scale", 1.0))
            dump_violations(
                df, args.label_col, args.step_col, x_cols, args.arrow_col,
                mask_vec, args.eps, args.eps_arrow,
                weights, pos_scale,
                args.violations_csv, args.top_k
            )
            print(f"[ok] wrote: {args.violations_csv} (top {args.top_k} forward-step cone violations)")

if __name__ == "__main__":
    main()
