#!/usr/bin/env python3
"""
Delta-cone signature screen (Lorentz test on step directions).

Tests cone-compatibility on step deltas only:

  Δx_t = x_{t+1} - x_t        (x excludes the arrow coordinate)
  arrow monotone: a_{t+1} >= a_t - eps
  cone condition: Q(Δx_t) <= eps      (sign convention: Q<=0 = "timelike or null")

Ranks masks/signatures primarily by worst-case cone satisfaction across labels.

Input CSV format:
  - required columns: label, iter
  - arrow coordinate: --arrow-col (default: v_arrow)
  - embedding coordinates x: default = all columns except label/iter/arrow, or specify --cols

Outputs:
  - delta_cone_signature_rank.csv (default) with per-mask scores

Example:
  python 28_delta_cone_signature_test.py \
      --embedding closure_embedding_per_step.csv \
      --arrow-col v_arrow \
      --require-nondegenerate \
      --min-cone-frac 0.95 \
      --eps 1e-12
"""

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signature:
    p: int
    q: int
    z: int

    def __str__(self) -> str:
        return f"({self.p},{self.q},{self.z})"


def mask_to_signature(mask: np.ndarray) -> Signature:
    p = int(np.sum(mask == +1))
    q = int(np.sum(mask == -1))
    z = int(np.sum(mask == 0))
    return Signature(p, q, z)


def Q_masked(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # x: (N,d), mask: (d,)
    return (x * x) @ mask


def per_label_steps(df: pd.DataFrame, x_cols: List[str], arrow_col: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns dict[label] = (X, A) where:
      X: (T,d) embedding coords excluding arrow
      A: (T,) arrow coordinate
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for lab, g in df.groupby("label"):
        gg = g.sort_values("iter")
        X = gg[x_cols].to_numpy(dtype=float)
        A = gg[arrow_col].to_numpy(dtype=float)
        out[str(lab)] = (X, A)
    return out


def score_delta_cone(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    mask: np.ndarray,
    eps: float,
) -> Dict[str, float]:
    """
    For each label trajectory (X,A):
      - arrow_ok[t] := A[t+1] >= A[t] - eps
      - cone_ok[t]  := Q(ΔX[t]) <= eps
      - ok[t]       := arrow_ok & cone_ok

    Aggregation across labels:
      - cone_frac_min  = min over labels of mean(ok)
      - cone_frac_mean = mean over labels of mean(ok)
      - max_Qd_violation_max/mean = max over labels of max(Q(ΔX)-eps)
    """
    cone_fracs: List[float] = []
    max_qd_viol: List[float] = []
    n_steps_total = 0

    for _, (X, A) in series.items():
        if X.shape[0] < 2:
            continue
        dX = X[1:] - X[:-1]  # (T-1,d)
        qd = Q_masked(dX, mask)  # (T-1,)
        qd_viol = qd - eps

        arrow_ok = (A[1:] >= A[:-1] - eps)
        cone_ok = (qd <= eps)

        ok = arrow_ok & cone_ok
        cone_fracs.append(float(np.mean(ok)))
        max_qd_viol.append(float(np.max(qd_viol)))
        n_steps_total += int(ok.size)

    if not cone_fracs:
        return {
            "cone_frac_min": float("nan"),
            "cone_frac_mean": float("nan"),
            "max_Qd_violation_max": float("nan"),
            "max_Qd_violation_mean": float("nan"),
            "n_labels": 0,
            "n_steps_total": 0,
        }

    return {
        "cone_frac_min": float(np.min(cone_fracs)),
        "cone_frac_mean": float(np.mean(cone_fracs)),
        "max_Qd_violation_max": float(np.max(max_qd_viol)),
        "max_Qd_violation_mean": float(np.mean(max_qd_viol)),
        "n_labels": int(len(cone_fracs)),
        "n_steps_total": int(n_steps_total),
    }


def enumerate_masks(d: int, allow_zero: bool) -> List[np.ndarray]:
    vals = [-1, +1] if not allow_zero else [-1, 0, +1]
    masks: List[np.ndarray] = []
    for tup in itertools.product(vals, repeat=d):
        m = np.array(tup, dtype=float)
        if np.all(m == 0):
            continue
        masks.append(m)
    return masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True, help="closure_embedding_per_step.csv")
    ap.add_argument("--out", default="delta_cone_signature_rank.csv")
    ap.add_argument("--arrow-col", default="v_arrow", help="column treated as arrow (monotone order parameter)")
    ap.add_argument("--cols", default="", help="comma-separated x embedding cols (EXCLUDES arrow). Default uses all except label/iter/arrow.")
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--allow-zero", action="store_true",
                    help="allow 0 entries in signature mask (degenerate). Default: disallow (z=0 only).")
    ap.add_argument("--require-nondegenerate", action="store_true",
                    help="filter to z==0 even if --allow-zero is set (recommended).")
    ap.add_argument("--min-cone-frac", type=float, default=0.95, help="filter threshold (min over labels).")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    for req in ("label", "iter"):
        if req not in df.columns:
            raise SystemExit("embedding must have columns: label, iter, ...")

    if args.arrow_col not in df.columns:
        raise SystemExit(f"--arrow-col {args.arrow_col} not found in CSV columns.")

    if args.cols.strip():
        x_cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        x_cols = [c for c in df.columns if c not in ("label", "iter", args.arrow_col)]

    # If user mistakenly included arrow in --cols, drop it (it's handled separately)
    x_cols = [c for c in x_cols if c != args.arrow_col]

    if len(x_cols) == 0:
        raise SystemExit("No x embedding columns selected (after excluding label/iter/arrow).")

    d = len(x_cols)

    series = per_label_steps(df, x_cols=x_cols, arrow_col=args.arrow_col)
    masks = enumerate_masks(d=d, allow_zero=args.allow_zero)

    rows = []
    for mask in masks:
        sig = mask_to_signature(mask)
        if (not args.allow_zero) and sig.z != 0:
            continue
        if args.require_nondegenerate and sig.z != 0:
            continue

        sc = score_delta_cone(series=series, mask=mask, eps=args.eps)
        rows.append({
            "p": sig.p, "q": sig.q, "z": sig.z,
            "mask": ",".join(str(int(v)) for v in mask.tolist()),
            **sc
        })

    out = pd.DataFrame(rows)

    # Rank: primary = cone_frac_min, then cone_frac_mean, then smallest violations.
    out = out.sort_values(
        by=["cone_frac_min", "cone_frac_mean", "max_Qd_violation_max", "max_Qd_violation_mean"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    out.to_csv(args.out, index=False)

    filt = out[out["cone_frac_min"] >= args.min_cone_frac]
    if len(filt) == 0:
        print("[warn] no signature passed thresholds.")
        print("[hint] try lowering --min-cone-frac, or use --allow-zero to inspect degenerates.")
    else:
        best = filt.iloc[0]
        print("=== Best delta-cone signature under constraints ===")
        print(best[["p","q","z","cone_frac_min","cone_frac_mean","max_Qd_violation_max","n_labels","n_steps_total"]].to_string())

    print(f"[ok] wrote: {args.out}")
    print(f"[info] x cols used: {x_cols}")
    print(f"[info] arrow col: {args.arrow_col}")
    print("[note] This script intentionally does NOT use Q(x) descent; it screens cone-compatibility on Δx only.")
    print("[note] Arrow is NOT included in Q; it is only used for monotonicity filtering.")


if __name__ == "__main__":
    main()
