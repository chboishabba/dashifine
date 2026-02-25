#!/usr/bin/env python3
"""
Reduced closure signature test for DASHI seam-cert pipeline.

Input:
  per_label_timeseries.csv
    expected columns: label, iter (or t), and beta columns b0..bK (at least b0..b3, ideally b0..b4)

We build a 4D closure embedding:
  phi_t = [ b0_t, b2_t, ||D_t||_1, t ]

Default projection P:
  - keep even beta coords (b0, b2, b4, ...)
  - zero odd beta coords (b1, b3, ...)

Thus defect D = x - P(x) = odd part (by default), and ||D||_1 = sum |odd coords|.

We then screen candidate "masked quadratic" forms:
  Q_s(x) = sum_i s_i * x_i^2, with s_i in {+1, -1, 0}

and test descent along the trajectory:
  Q(x_{t+1}) <= Q(x_t) + eps

Outputs:
  - signature_rank_reduced4d.csv
  - signature_rank_reduced4d_raw.csv
  - best_masks_per_signature.csv
"""

import argparse
import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Mask:
    """Vector of signs in {-1,0,+1} for 4D embedding."""
    s: Tuple[int, int, int, int]

    def __str__(self) -> str:
        return "".join({-1: "-", 0: "0", 1: "+"}[x] for x in self.s)

    @property
    def p_q_z(self) -> Tuple[int, int, int]:
        p = sum(1 for x in self.s if x == 1)
        q = sum(1 for x in self.s if x == -1)
        z = sum(1 for x in self.s if x == 0)
        return (p, q, z)


def masked_quadratic(x: np.ndarray, mask: Mask) -> np.ndarray:
    """
    x: shape (T,4)
    returns Q(x) shape (T,)
    """
    s = np.array(mask.s, dtype=float)  # -1,0,+1
    return (x * x) @ s


def descent_stats(q: np.ndarray, eps: float) -> Tuple[float, float, int]:
    """
    For a sequence q_t, test q_{t+1} <= q_t + eps.
    Return:
      fraction_satisfied, max_violation, first_violation_iter (1-based index of t+1), or -1 if none
    """
    if q.shape[0] < 2:
        return (1.0, 0.0, -1)
    dq = q[1:] - q[:-1]
    viol = dq - eps
    ok = viol <= 0.0
    frac = float(np.mean(ok)) if ok.size else 1.0
    maxv = float(np.max(viol)) if viol.size else 0.0
    if np.any(~ok):
        first = int(np.argmax(~ok)) + 1  # index of q_{t+1} in the sequence (0-based +1)
    else:
        first = -1
    return (frac, maxv, first)


def all_masks_for_signature(p: int, q: int, z: int) -> List[Mask]:
    """
    Enumerate all distinct sign masks in 4D with exactly:
      p positives, q negatives, z zeros
    """
    assert p + q + z == 4
    base = [1] * p + [-1] * q + [0] * z
    masks = set(itertools.permutations(base, 4))
    return [Mask(tuple(m)) for m in sorted(masks)]


def build_embedding(df_label: pd.DataFrame, beta_cols: List[str], t_col: str) -> np.ndarray:
    """
    Build (T,4) embedding: [b0, b2, ||odd||_1, t]
    """
    df_label = df_label.sort_values(t_col).reset_index(drop=True)
    b0 = df_label["b0"].to_numpy(dtype=float)
    if "b2" not in df_label.columns:
        raise ValueError("Need column b2 to build reduced embedding.")
    b2 = df_label["b2"].to_numpy(dtype=float)

    # Defect norm = sum |odd beta coords|
    odd_cols = [c for c in beta_cols if (c.startswith("b") and c[1:].isdigit() and (int(c[1:]) % 2 == 1))]
    if not odd_cols:
        # if no odd columns exist, defect is zero
        defect = np.zeros_like(b0)
    else:
        defect = np.sum(np.abs(df_label[odd_cols].to_numpy(dtype=float)), axis=1)

    t = df_label[t_col].to_numpy(dtype=float)
    x = np.stack([b0, b2, defect, t], axis=1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True, help="Path to per_label_timeseries.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--eps", type=float, default=1e-12, help="Descent tolerance")
    ap.add_argument("--exclude_degenerate", action="store_true", default=True,
                    help="Exclude p=q=0 (zero quadratic) signatures (default True).")
    ap.add_argument("--no_exclude_degenerate", dest="exclude_degenerate", action="store_false",
                    help="Allow degenerate signature p=q=0 (not recommended).")

    # Candidate signatures to test in 4D: list of (p,q,z) with p+q+z=4
    ap.add_argument("--signatures", default="3,1,0;1,3,0;2,2,0;4,0,0;0,4,0;3,0,1;0,3,1;2,1,1;1,2,1;2,0,2;0,2,2;1,1,2",
                    help="Semicolon-separated list of p,q,z triplets (sum to 4), e.g. '3,1,0;2,2,0;4,0,0'")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.timeseries)

    # Find time column
    t_col = None
    for cand in ["iter", "t", "T_iter", "step"]:
        if cand in df.columns:
            t_col = cand
            break
    if t_col is None:
        raise ValueError("Couldn't find time column. Expected one of: iter, t, T_iter, step")

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in timeseries CSV.")

    # beta columns
    beta_cols = [c for c in df.columns if c.startswith("b") and (len(c) > 1) and c[1:].isdigit()]
    if "b0" not in df.columns:
        raise ValueError("Need b0 column.")
    if "b2" not in df.columns:
        raise ValueError("Need b2 column.")
    if not beta_cols:
        raise ValueError("No beta columns found (b0,b1,...)")

    # Parse signature list
    sigs: List[Tuple[int, int, int]] = []
    for item in args.signatures.split(";"):
        item = item.strip()
        if not item:
            continue
        p, q, z = (int(x.strip()) for x in item.split(","))
        if p + q + z != 4:
            raise ValueError(f"Signature {p,q,z} does not sum to 4.")
        if args.exclude_degenerate and p == 0 and q == 0:
            continue
        sigs.append((p, q, z))
    sigs = list(dict.fromkeys(sigs))  # unique preserve order

    # Evaluate per label
    rows_raw = []
    rows_best = []

    for label, dfl in df.groupby("label"):
        x = build_embedding(dfl, beta_cols=beta_cols, t_col=t_col)  # (T,4)

        for (p, q, z) in sigs:
            masks = all_masks_for_signature(p, q, z)

            best = None
            for m in masks:
                qx = masked_quadratic(x, m)
                frac, maxv, first = descent_stats(qx, eps=args.eps)

                rows_raw.append({
                    "label": label,
                    "p": p, "q": q, "z": z,
                    "mask": str(m),
                    "descent_frac": frac,
                    "max_violation": maxv,
                    "first_violation_iter": first,
                })

                # pick best by: max descent_frac, then min max_violation
                score = (frac, -maxv)
                if best is None or score > best[0]:
                    best = (score, m, frac, maxv, first)

            assert best is not None
            _, best_mask, best_frac, best_maxv, best_first = best
            rows_best.append({
                "label": label,
                "p": p, "q": q, "z": z,
                "best_mask": str(best_mask),
                "best_descent_frac": best_frac,
                "best_max_violation": best_maxv,
                "best_first_violation_iter": best_first,
            })

    df_raw = pd.DataFrame(rows_raw)
    df_best = pd.DataFrame(rows_best)

    # Aggregate ranking across labels
    # Primary: min over labels of best_descent_frac (worst-case guarantee)
    # Secondary: max over labels of best_max_violation (worst-case violation size)
    agg = []
    for (p, q, z), g in df_best.groupby(["p", "q", "z"]):
        agg.append({
            "p": int(p), "q": int(q), "z": int(z),
            "n_labels": int(g.shape[0]),
            "best_descent_frac_min": float(g["best_descent_frac"].min()),
            "best_descent_frac_mean": float(g["best_descent_frac"].mean()),
            "best_max_violation_max": float(g["best_max_violation"].max()),
            "best_max_violation_mean": float(g["best_max_violation"].mean()),
        })
    df_rank = pd.DataFrame(agg)

    # Rank: highest min descent frac, then lowest max violation
    df_rank = df_rank.sort_values(
        by=["best_descent_frac_min", "best_max_violation_max", "best_descent_frac_mean", "best_max_violation_mean"],
        ascending=[False, True, False, True]
    ).reset_index(drop=True)

    # Write outputs
    out_rank = os.path.join(args.outdir, "signature_rank_reduced4d.csv")
    out_raw = os.path.join(args.outdir, "signature_rank_reduced4d_raw.csv")
    out_best = os.path.join(args.outdir, "best_masks_per_signature.csv")

    df_rank.to_csv(out_rank, index=False)
    df_raw.to_csv(out_raw, index=False)
    df_best.to_csv(out_best, index=False)

    # Print top results
    print("[ok] wrote:", out_rank)
    print("[ok] wrote:", out_best)
    print("[ok] wrote:", out_raw)
    print()
    print("=== Top signatures (reduced 4D embedding) ===")
    print(df_rank.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
