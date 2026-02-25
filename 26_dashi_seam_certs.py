#!/usr/bin/env python3
"""
dashi_seam_certs.py

Compute DASHI seam-certificates from per_label_timeseries.csv:

- Exact MDL-proxy descent:  E_MDL_proxy(t+1) <= E_MDL_proxy(t)
- Fejér-to-fixed-set:       d(Px, y) <= d(x, y)   for many random y in Fix(P)
- ClosestPoint / prox:      d(x, Px) <= d(x, y)   for many random y in Fix(P)
- Translation invariance:   d(x+z, y+z) == d(x,y)
- Defect monotonicity:      ||D(x_{t+1})|| <= ||D(x_t)||  with D(x)=x-Px
- MaskedQuadratic screening by candidate signatures (p,q,z): Q nonincrease + Fejér(Q)

Assumptions:
- Beta coordinates are the b* columns (b0,b1,...).
- P is "even projector": keep even indices, zero odd indices.
- Distances: L1, L2; and "MDL-distance" is L1 by default (translation invariant).
  (You can swap d_mdl to something else if you have a true translation-invariant MDL distance.)

Outputs:
- outdir/mdl_descent_exact.csv
- outdir/fejer_set_report.csv
- outdir/closestpoint_report.csv
- outdir/transinv_report.csv
- outdir/defect_monotonicity_report.csv
- outdir/maskedQ_signature_rank_raw.csv
- outdir/maskedQ_signature_rank.csv
- outdir/overall_certification.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


EPS_DEFAULT = 1e-12


def find_beta_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("b") and c[1:].isdigit()]
    cols_sorted = sorted(cols, key=lambda c: int(c[1:]))
    if not cols_sorted:
        raise ValueError("No beta columns found (expected b0,b1,...)")
    return cols_sorted


def P_even(x: np.ndarray) -> np.ndarray:
    """Projection: keep even indices, zero odd indices."""
    y = x.copy()
    y[1::2] = 0.0
    return y


def D_defect(x: np.ndarray) -> np.ndarray:
    return x - P_even(x)


def d_L1(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.abs(x - y)))


def d_L2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))


def d_MDL_dist(x: np.ndarray, y: np.ndarray) -> float:
    """
    Translation-invariant 'MDL-distance' placeholder.

    In practice, your MDL proxy is an energy E(x), not a metric d(x,y).
    For ClosestPoint/Fejér seams you want a genuine translation-invariant distance;
    L1 is the clean choice and is what your seam-certs already support.
    """
    return d_L1(x, y)


def sample_fixpoint_y(rng: np.random.Generator, dim: int, scale: float = 1.0) -> np.ndarray:
    """
    Sample y in Fix(P) by sampling a random vector then projecting:
      y := P(z) so P(y)=y.
    """
    z = rng.normal(0.0, scale, size=(dim,))
    return P_even(z)


def masked_quadratic_Q(x: np.ndarray, signs: np.ndarray) -> float:
    """
    Q(x) = sum_i s_i * x_i^2 where s_i in {+1, -1, 0}.
    """
    return float(np.sum(signs * (x * x)))


def enumerate_signatures(n: int) -> List[Tuple[int, int, int, np.ndarray]]:
    """
    All (p,q,z) with p+q+z=n; build a diagonal sign vector:
      first p entries +1, next q entries -1, next z entries 0.
    NOTE: Ordering matters; we keep a canonical ordering for screening.
    """
    out = []
    for p in range(n + 1):
        for q in range(n - p + 1):
            z = n - p - q
            signs = np.zeros((n,), dtype=float)
            signs[:p] = 1.0
            signs[p : p + q] = -1.0
            # remaining already 0
            out.append((p, q, z, signs))
    return out


@dataclass
class FejerResult:
    frac: float
    max_violation: float
    argmax_iter: int


def fejer_nonincrease_over_time(
    series: np.ndarray,  # shape (T, dim)
    energy_fn,
    eps: float,
) -> FejerResult:
    """
    Check energy nonincrease across consecutive steps:
      E(x_{t+1}) <= E(x_t) + eps
    Returns fraction satisfied, maximum violation, and iteration of worst violation (1..T-1).
    """
    T = series.shape[0]
    ok = 0
    max_viol = -1e300
    argmax = -1
    for t in range(T - 1):
        e0 = energy_fn(series[t])
        e1 = energy_fn(series[t + 1])
        viol = e1 - e0
        if viol <= eps:
            ok += 1
        if viol > max_viol:
            max_viol = viol
            argmax = t + 1
    frac = ok / max(1, (T - 1))
    return FejerResult(frac=frac, max_violation=float(max_viol), argmax_iter=int(argmax))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True, help="Path to per_label_timeseries.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT)
    ap.add_argument("--n_y", type=int, default=2000, help="Trials per label for Fejér/ClosestPoint")
    ap.add_argument("--n_transinv", type=int, default=5000, help="Translation invariance trials per label+metric")
    ap.add_argument("--y_scale", type=float, default=1.0, help="Stddev for Fix(P) sampling")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.timeseries)
    beta_cols = find_beta_cols(df)
    dim = len(beta_cols)

    # group by label
    labels = sorted(df["label"].unique().tolist())

    # ----------------------------
    # 1) Exact MDL-proxy descent
    # ----------------------------
    if "E_MDL_proxy" not in df.columns:
        raise ValueError("Expected E_MDL_proxy in timeseries; not found.")

    mdl_rows = []
    for lab in labels:
        g = df[df["label"] == lab].sort_values("iter")
        E = g["E_MDL_proxy"].to_numpy(dtype=float)
        violations = []
        worst_inc = -1e300
        worst_iter = -1
        for i in range(len(E) - 1):
            inc = E[i + 1] - E[i]
            if inc > args.eps:
                violations.append((i + 1, inc))
            if inc > worst_inc:
                worst_inc = inc
                worst_iter = int(g["iter"].iloc[i + 1])
        mdl_rows.append(
            dict(
                label=lab,
                T_iter=int(g["iter"].max()),
                MDL_monotone=(len(violations) == 0),
                MDL_violations=len(violations),
                MDL_worst_increase=float(worst_inc),
                MDL_worst_iter=int(worst_iter),
            )
        )
    mdl_df = pd.DataFrame(mdl_rows)
    mdl_df.to_csv(os.path.join(args.outdir, "mdl_descent_exact.csv"), index=False)

    # ----------------------------
    # 2) Fejér-to-fixed-set
    #    and 3) ClosestPoint
    # ----------------------------
    def dist_fn(metric: str):
        if metric == "L1":
            return d_L1
        if metric == "L2":
            return d_L2
        if metric == "MDL":
            return d_MDL_dist
        raise ValueError(metric)

    fejer_set_rows = []
    closest_rows = []

    for lab in labels:
        g = df[df["label"] == lab].sort_values("iter")
        X = g[beta_cols].to_numpy(dtype=float)  # (T,dim)
        T_iter = int(g["iter"].max())

        # We'll test at every timepoint x_t (including t=T) vs random y in Fix(P).
        for metric in ["L1", "L2", "MDL"]:
            d = dist_fn(metric)

            n_ok_fejer = 0
            n_ok_closest = 0
            total = 0

            worst_fejer = -1e300
            worst_fejer_arg = -1
            worst_closest = -1e300
            worst_closest_arg = -1

            for t in range(X.shape[0]):
                x = X[t]
                px = P_even(x)

                # distance to projection (for closestpoint)
                dxpx = d(x, px)

                # sample multiple y for each x_t
                for k in range(max(1, args.n_y // X.shape[0])):
                    y = sample_fixpoint_y(rng, dim=dim, scale=args.y_scale)

                    lhs_fejer = d(px, y)
                    rhs_fejer = d(x, y)
                    viol_fejer = lhs_fejer - rhs_fejer

                    lhs_closest = dxpx
                    rhs_closest = d(x, y)
                    viol_closest = lhs_closest - rhs_closest

                    total += 1
                    if viol_fejer <= args.eps:
                        n_ok_fejer += 1
                    if viol_closest <= args.eps:
                        n_ok_closest += 1

                    if viol_fejer > worst_fejer:
                        worst_fejer = viol_fejer
                        worst_fejer_arg = int(g["iter"].iloc[t])
                    if viol_closest > worst_closest:
                        worst_closest = viol_closest
                        worst_closest_arg = int(g["iter"].iloc[t])

            fejer_set_rows.append(
                dict(
                    label=lab,
                    metric=metric,
                    fejer_set_frac=n_ok_fejer / max(1, total),
                    fejer_set_max_violation=float(worst_fejer),
                    fejer_set_argmax=int(worst_fejer_arg),
                    T_iter=T_iter,
                    beta_dim=dim,
                )
            )
            closest_rows.append(
                dict(
                    label=lab,
                    metric=metric,
                    closest_frac=n_ok_closest / max(1, total),
                    closest_max_violation=float(worst_closest),
                    closest_argmax=int(worst_closest_arg),
                    T_iter=T_iter,
                    beta_dim=dim,
                )
            )

    fejer_set_df = pd.DataFrame(fejer_set_rows)
    fejer_set_df.to_csv(os.path.join(args.outdir, "fejer_set_report.csv"), index=False)

    closest_df = pd.DataFrame(closest_rows)
    closest_df.to_csv(os.path.join(args.outdir, "closestpoint_report.csv"), index=False)

    # ----------------------------
    # 4) Translation invariance
    # ----------------------------
    trans_rows = []
    for lab in labels:
        g = df[df["label"] == lab].sort_values("iter")
        X = g[beta_cols].to_numpy(dtype=float)
        # pick a few random pairs (x,y) from the timepoints, plus random z
        for metric in ["L1", "L2", "MDL"]:
            d = dist_fn(metric)
            ok = 0
            worst = 0.0
            for _ in range(args.n_transinv):
                i = int(rng.integers(0, X.shape[0]))
                j = int(rng.integers(0, X.shape[0]))
                x = X[i]
                y = X[j]
                z = rng.normal(0.0, 1.0, size=(dim,))
                lhs = d(x + z, y + z)
                rhs = d(x, y)
                err = abs(lhs - rhs)
                if err <= 1e-10:
                    ok += 1
                worst = max(worst, err)
            trans_rows.append(
                dict(
                    label=lab,
                    metric=metric,
                    transinv_frac=ok / max(1, args.n_transinv),
                    worst_abs_err=float(worst),
                    n_trials=int(args.n_transinv),
                )
            )
    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(os.path.join(args.outdir, "transinv_report.csv"), index=False)

    # ----------------------------
    # 5) Defect monotonicity
    # ----------------------------
    defect_rows = []
    for lab in labels:
        g = df[df["label"] == lab].sort_values("iter")
        X = g[beta_cols].to_numpy(dtype=float)
        D = np.stack([D_defect(x) for x in X], axis=0)
        D1 = np.sum(np.abs(D), axis=1)
        D2 = np.linalg.norm(D, axis=1)

        def monotone(arr: np.ndarray) -> Tuple[bool, int, float]:
            viol = 0
            worst = -1e300
            for t in range(len(arr) - 1):
                inc = arr[t + 1] - arr[t]
                if inc > args.eps:
                    viol += 1
                worst = max(worst, inc)
            return (viol == 0), viol, float(worst)

        m1, v1, w1 = monotone(D1)
        m2, v2, w2 = monotone(D2)
        defect_rows.append(
            dict(
                label=lab,
                T_iter=int(g["iter"].max()),
                beta_dim=dim,
                L1_monotone=bool(m1),
                L1_violations=int(v1),
                L1_worst_increase=float(w1),
                L2_monotone=bool(m2),
                L2_violations=int(v2),
                L2_worst_increase=float(w2),
            )
        )
    defect_df = pd.DataFrame(defect_rows)
    defect_df.to_csv(os.path.join(args.outdir, "defect_monotonicity_report.csv"), index=False)

    # ----------------------------
    # 6) MaskedQuadratic signature screening
    # ----------------------------
    sigs = enumerate_signatures(dim)
    sig_rows = []
    for lab in labels:
        g = df[df["label"] == lab].sort_values("iter")
        X = g[beta_cols].to_numpy(dtype=float)

        for (p, q, z, signs) in sigs:
            # Q nonincrease across time
            def E_Q(v: np.ndarray) -> float:
                return masked_quadratic_Q(v, signs)

            res = fejer_nonincrease_over_time(X, E_Q, eps=args.eps)
            sig_rows.append(
                dict(
                    label=lab,
                    p=p,
                    q=q,
                    z=z,
                    fejer_frac_Q=res.frac,
                    max_violation_Q=res.max_violation,
                    argmax_iter_Q=res.argmax_iter,
                    use_abs=False,
                )
            )

    sig_raw_df = pd.DataFrame(sig_rows)
    sig_raw_df.to_csv(os.path.join(args.outdir, "maskedQ_signature_rank_raw.csv"), index=False)

    # pick best per label: highest fejer_frac_Q then smallest max_violation_Q
    best_rows = []
    for lab in labels:
        h = sig_raw_df[sig_raw_df["label"] == lab].copy()
        h = h.sort_values(
            by=["fejer_frac_Q", "max_violation_Q", "p", "q", "z"],
            ascending=[False, True, True, True, True],
        )
        best_rows.append(h.iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(args.outdir, "maskedQ_signature_rank.csv"), index=False)

    # ----------------------------
    # 7) Overall rollup
    # ----------------------------
    overall = {
        "beta_dim": dim,
        "labels": labels,
        "mdl_monotone_descent_all_labels": bool(mdl_df["MDL_monotone"].all()),
        "fejer_set_L1_min": float(fejer_set_df[fejer_set_df["metric"] == "L1"]["fejer_set_frac"].min()),
        "closest_L1_min": float(closest_df[closest_df["metric"] == "L1"]["closest_frac"].min()),
        "transinv_L1_min": float(trans_df[trans_df["metric"] == "L1"]["transinv_frac"].min()),
        "defect_L1_all_labels": bool(defect_df["L1_monotone"].all()),
        "best_maskedQ_signatures": best_df[["label", "p", "q", "z", "fejer_frac_Q", "max_violation_Q"]].to_dict(orient="records"),
        "eps": args.eps,
        "seed": args.seed,
        "n_y": args.n_y,
        "n_transinv": args.n_transinv,
    }
    with open(os.path.join(args.outdir, "overall_certification.json"), "w") as f:
        json.dump(overall, f, indent=2, sort_keys=True)

    print("[ok] wrote:",
          os.path.join(args.outdir, "mdl_descent_exact.csv"))
    print("[ok] wrote:",
          os.path.join(args.outdir, "fejer_set_report.csv"))
    print("[ok] wrote:",
          os.path.join(args.outdir, "closestpoint_report.csv"))
    print("[ok] wrote:",
          os.path.join(args.outdir, "transinv_report.csv"))
    print("[ok] wrote:",
          os.path.join(args.outdir, "defect_monotonicity_report.csv"))
    print("[ok] wrote:",
          os.path.join(args.outdir, "maskedQ_signature_rank_raw.csv"))
    print("[ok] wrote:",
          os.path.join(args.outdir, "maskedQ_signature_rank.csv"))
    print("[ok] wrote:",
          os.path.join(args.outdir, "overall_certification.json"))


if __name__ == "__main__":
    main()
