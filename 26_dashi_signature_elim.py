#!/usr/bin/env python3
import argparse
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# --------- utilities ---------

def vcols(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def safe_log(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    return np.log(np.maximum(x, eps))


@dataclass(frozen=True)
class Signature:
    p: int
    q: int
    z: int

    def __str__(self) -> str:
        return f"({self.p},{self.q},{self.z})"


def mask_to_signature(mask: np.ndarray) -> Signature:
    # mask entries in {-1, 0, +1}
    p = int(np.sum(mask == +1))
    q = int(np.sum(mask == -1))
    z = int(np.sum(mask == 0))
    return Signature(p, q, z)


def Q_masked(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # x shape: (N, d), mask shape: (d,)
    # Q(x) = Σ mask_i * x_i^2
    return (x * x) @ mask


# --------- scoring ---------

def per_label_steps(df: pd.DataFrame, cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns dict label -> array shape (T, d) sorted by iter.
    Assumes df has columns: label, iter, cols...
    """
    out = {}
    for lab, g in df.groupby("label"):
        gg = g.sort_values("iter")
        out[lab] = gg[cols].to_numpy(dtype=float)
    return out


def score_signature(
    series: Dict[str, np.ndarray],
    mask: np.ndarray,
    arrow_idx: int,
    cone_use_delta: bool,
    eps: float,
) -> Dict[str, float]:
    """
    Computes:
      - descent_frac_min/mean: fraction of steps with Q(next) <= Q(cur)+eps
      - max_violation_max/mean: max over steps of (Q(next)-Q(cur))  (positive = violation)
      - cone_frac_min/mean: fraction of steps with arrow(next) >= arrow(cur) AND (cone condition)
        cone condition default: Q(delta) <= eps (timelike-or-null if you take Q<=0 as cone)
    """
    descent_fracs = []
    max_violations = []
    cone_fracs = []

    for lab, X in series.items():
        if X.shape[0] < 2:
            continue
        Qx = Q_masked(X, mask)
        dQ = Qx[1:] - Qx[:-1]

        descent = (dQ <= eps)
        descent_frac = float(np.mean(descent))
        descent_fracs.append(descent_frac)
        max_violations.append(float(np.max(dQ)))

        # Arrow monotone premise:
        arrow = X[:, arrow_idx]
        arrow_ok = (arrow[1:] >= arrow[:-1] - eps)

        if cone_use_delta:
            dX = X[1:] - X[:-1]
            Qd = Q_masked(dX, mask)
            cone_ok = (Qd <= eps)
        else:
            # alternate: require next itself in cone wrt origin
            cone_ok = (Qx[1:] <= eps)

        cone_frac = float(np.mean(arrow_ok & cone_ok))
        cone_fracs.append(cone_frac)

    # aggregate across labels
    def agg(fracs: List[float]) -> Tuple[float, float]:
        if not fracs:
            return (float("nan"), float("nan"))
        return (float(np.min(fracs)), float(np.mean(fracs)))

    descent_min, descent_mean = agg(descent_fracs)
    cone_min, cone_mean = agg(cone_fracs)
    maxv_max = float(np.max(max_violations)) if max_violations else float("nan")
    maxv_mean = float(np.mean(max_violations)) if max_violations else float("nan")

    return {
        "descent_frac_min": descent_min,
        "descent_frac_mean": descent_mean,
        "max_violation_max": maxv_max,
        "max_violation_mean": maxv_mean,
        "cone_frac_min": cone_min,
        "cone_frac_mean": cone_mean,
        "n_labels": int(len(descent_fracs)),
    }


def enumerate_masks(d: int, allow_zero: bool) -> List[np.ndarray]:
    vals = [-1, +1] if not allow_zero else [-1, 0, +1]
    masks = []
    for tup in itertools.product(vals, repeat=d):
        m = np.array(tup, dtype=float)
        # ignore all-zero mask (Q==0 everywhere)
        if np.all(m == 0):
            continue
        masks.append(m)
    return masks


# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True, help="closure_embedding_per_step.csv")
    ap.add_argument("--out", default="signature_elim_rank.csv")
    ap.add_argument("--arrow-col", default="v_arrow", help="column in embedding to treat as arrow")
    ap.add_argument("--cols", default="", help="comma-separated embedding cols; default auto v_* except label/iter")
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--allow-zero", action="store_true",
                    help="allow 0 entries in signature mask (degenerate). Default: disallow (z=0 only).")
    ap.add_argument("--cone-use-delta", action="store_true",
                    help="cone test uses Q(delta) <= 0 (recommended). If off, uses Q(next)<=0.")
    ap.add_argument("--require-nondegenerate", action="store_true",
                    help="explicitly filter to z==0 even if --allow-zero is set.")
    ap.add_argument("--min-cone-frac", type=float, default=0.95, help="filter threshold (min over labels).")
    ap.add_argument("--min-descent-frac", type=float, default=0.95, help="filter threshold (min over labels).")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    if "label" not in df.columns or "iter" not in df.columns:
        raise SystemExit("embedding must have columns: label, iter, ...")

    if args.cols.strip():
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        cols = [c for c in df.columns if c not in ("label", "iter")]

    if args.arrow_col not in cols:
        raise SystemExit(f"--arrow-col {args.arrow_col} must be included in --cols (or present by default).")

    # Keep deterministic column order
    cols = list(cols)
    d = len(cols)
    arrow_idx = cols.index(args.arrow_col)

    series = per_label_steps(df, cols)

    masks = enumerate_masks(d=d, allow_zero=args.allow_zero)
    rows = []

    for mask in masks:
        sig = mask_to_signature(mask)
        if (not args.allow_zero) and sig.z != 0:
            continue
        if args.require_nondegenerate and sig.z != 0:
            continue

        sc = score_signature(
            series=series,
            mask=mask,
            arrow_idx=arrow_idx,
            cone_use_delta=args.cone_use_delta,
            eps=args.eps,
        )
        rows.append({
            "p": sig.p, "q": sig.q, "z": sig.z,
            "mask": ",".join(str(int(v)) for v in mask.tolist()),
            **sc
        })

    out = pd.DataFrame(rows)
    # rank: prefer high descent_min then high cone_min then low max_violation_max
    out = out.sort_values(
        by=["descent_frac_min", "cone_frac_min", "max_violation_max", "descent_frac_mean", "cone_frac_mean"],
        ascending=[False, False, True, False, False],
    )

    out.to_csv(args.out, index=False)

    # pick "best" under thresholds (this is the seam certificate you can point Agda at)
    filt = out[
        (out["descent_frac_min"] >= args.min_descent_frac) &
        (out["cone_frac_min"] >= args.min_cone_frac)
    ]
    if len(filt) == 0:
        print("[warn] no signature passed thresholds.")
        print("[hint] try lowering --min-cone-frac / --min-descent-frac or use --allow-zero to inspect degenerates.")
    else:
        best = filt.iloc[0]
        print("=== Best signature under constraints ===")
        print(best[["p","q","z","descent_frac_min","cone_frac_min","max_violation_max","n_labels"]].to_string())

    print(f"[ok] wrote: {args.out}")
    print(f"[info] cols used: {cols}")
    print("[note] If degenerates dominate, add: --require-nondegenerate and keep cone test with --cone-use-delta.")


if __name__ == "__main__":
    main()
