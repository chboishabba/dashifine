#!/usr/bin/env python3
"""
hepdata_closure_tests.py

One-stop “seam-certificate” runner + 4D closure embedding + signature elimination.

Inputs:
  - per_label_timeseries.csv   (your iteration-by-iteration table; must include beta columns)
Optional:
  - any extra energy columns you already compute (MDL proxy etc). If absent, we compute a default MDL proxy.

Outputs (CSV) in --out:
  - defect_monotonicity_report.csv
  - mdl_descent_exact.csv
  - fejer_set_report.csv
  - closestpoint_report.csv
  - maskedQ_signature_rank.csv
  - closure_embedding_per_step.csv

Typical usage:
  python hepdata_closure_tests.py \
    --timeseries hepdata_mdl_fejer_timeseries.csv \
    --out closure_out \
    --beta-cols b0 b1 b2 b3 b4 \
    --P-odd-indices 1 3 \
    --mdlname E_MDL_proxy \
    --chi2name chi2_dof \
    --alpha-name alpha \
    --iters-max 999 \
    --y-samples 64 \
    --seed 0

Notes:
  - We treat P as “kill odd indices” by default (Fix(P) = {b_odd=0}).
  - We build a reduced 4D “closure embedding” per step:
        v = [ ||Pβ||2 , ||Dβ||2 , depth , arrow ]
    where:
        depth = log10(alpha) if present else iter
        arrow = MDL proxy if present else -log(chi2_dof) (clipped)
  - Fejér-to-fixed-set is checked in L1 and in the MDL-proxy distance (if available).
  - ClosestPoint is tested empirically:
        d(x, Px) <= d(x, y) for random y ∈ Fix(P)
    with y drawn from (empirical distribution of Px) + small even-only noise.
  - Signature elimination: ranks candidate (p,q,0) masks by “Fejér fraction” for Q on the 4D embedding.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))

def l1(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sum(np.abs(x), axis=axis)

def l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=axis))

def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

# -------------------------
# Projection / defect
# -------------------------

@dataclass
class ProjectionDefect:
    odd_idx: List[int]  # indices to zero out
    beta_dim: int

    def P(self, beta: np.ndarray) -> np.ndarray:
        """Projection: zero odd indices."""
        out = np.array(beta, copy=True)
        out[..., self.odd_idx] = 0.0
        return out

    def D(self, beta: np.ndarray) -> np.ndarray:
        """Defect: x - P(x)."""
        return beta - self.P(beta)

# -------------------------
# Distances / energies
# -------------------------

def dist_L1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))

def dist_L2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.sum(d * d)))

def dist_MDL_proxy(a: np.ndarray, b: np.ndarray, mdl_scale: float = 1.0) -> float:
    """
    A *distance-like* proxy built from L1 with a scale.
    If you already have a better MDL-distance, plug it in here.
    """
    return mdl_scale * dist_L1(a, b)

# -------------------------
# Fix(P) sampling
# -------------------------

def sample_y_in_fixP(
    rng: np.random.Generator,
    Px_pool: np.ndarray,
    even_idx: List[int],
    n: int,
    noise_sigma: float = 0.05,
) -> np.ndarray:
    """
    Sample y ∈ Fix(P) by:
      y = Px_pool[random] + even-only Gaussian noise
    """
    m = Px_pool.shape[0]
    idx = rng.integers(0, m, size=n)
    y = Px_pool[idx].copy()
    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, size=y.shape)
        # only allow noise on even indices (Fix(P) preserved)
        mask = np.zeros(y.shape[-1], dtype=bool)
        mask[even_idx] = True
        y[..., mask] += noise[..., mask]
        # enforce exact fix
        odd_mask = ~mask
        y[..., odd_mask] = 0.0
    return y

# -------------------------
# Fejér-to-fixed-set
# -------------------------

def fejer_set_check(
    betas: np.ndarray,
    P: ProjectionDefect,
    y_samples: np.ndarray,
    metric: str = "L1",
    mdl_scale: float = 1.0,
) -> Tuple[float, float, int]:
    """
    Check Fejér-to-fixed-set:
        d(Px, y) <= d(x, y)   for many y in Fix(P)
    across all x_t (except last) and many y.

    Returns:
      (fraction_satisfied, max_violation, argmax_flat_index)
    """
    Px = P.P(betas)
    T = betas.shape[0]

    sat = 0
    tot = 0
    max_viol = -np.inf
    argmax = -1

    def d(a, b) -> float:
        if metric == "L1":
            return dist_L1(a, b)
        if metric == "L2":
            return dist_L2(a, b)
        if metric == "MDL":
            return dist_MDL_proxy(a, b, mdl_scale=mdl_scale)
        raise ValueError(f"Unknown metric: {metric}")

    # compare stepwise x_t (not x_{t+1}) because statement is per x:
    # Fejér says projection doesn't increase distance to Fix-set.
    # So test for each t: d(P x_t, y) <= d(x_t, y).
    for t in range(T):
        x = betas[t]
        px = Px[t]
        for j in range(y_samples.shape[0]):
            y = y_samples[j]
            lhs = d(px, y)
            rhs = d(x, y)
            viol = lhs - rhs
            if lhs <= rhs + 1e-12:
                sat += 1
            if viol > max_viol:
                max_viol = viol
                argmax = t * y_samples.shape[0] + j
            tot += 1

    frac = sat / max(tot, 1)
    return float(frac), float(max_viol), int(argmax)

# -------------------------
# Closest point (empirical)
# -------------------------

def closest_point_check(
    betas: np.ndarray,
    P: ProjectionDefect,
    y_samples: np.ndarray,
    metric: str = "L1",
    mdl_scale: float = 1.0,
) -> Tuple[float, float, int]:
    """
    Check empirical closest-point:
        d(x, P x) <= d(x, y)  for random y ∈ Fix(P)
    across all x and y.

    Returns:
      (fraction_satisfied, max_violation, argmax_flat_index)
    """
    Px = P.P(betas)
    T = betas.shape[0]

    sat = 0
    tot = 0
    max_viol = -np.inf
    argmax = -1

    def d(a, b) -> float:
        if metric == "L1":
            return dist_L1(a, b)
        if metric == "L2":
            return dist_L2(a, b)
        if metric == "MDL":
            return dist_MDL_proxy(a, b, mdl_scale=mdl_scale)
        raise ValueError(f"Unknown metric: {metric}")

    for t in range(T):
        x = betas[t]
        px = Px[t]
        d_x_px = d(x, px)
        for j in range(y_samples.shape[0]):
            y = y_samples[j]
            d_x_y = d(x, y)
            viol = d_x_px - d_x_y
            if d_x_px <= d_x_y + 1e-12:
                sat += 1
            if viol > max_viol:
                max_viol = viol
                argmax = t * y_samples.shape[0] + j
            tot += 1

    frac = sat / max(tot, 1)
    return float(frac), float(max_viol), int(argmax)

# -------------------------
# Defect monotonicity
# -------------------------

def defect_monotonicity(betas: np.ndarray, P: ProjectionDefect) -> Dict[str, object]:
    D = P.D(betas)
    dL1 = l1(D, axis=-1)
    dL2 = l2(D, axis=-1)
    # monotone nonincreasing across t
    inc_L1 = dL1[1:] - dL1[:-1]
    inc_L2 = dL2[1:] - dL2[:-1]
    return {
        "L1_monotone": bool(np.all(inc_L1 <= 1e-12)),
        "L1_violations": int(np.sum(inc_L1 > 1e-12)),
        "L1_worst_increase": float(np.max(inc_L1)) if inc_L1.size else 0.0,
        "L2_monotone": bool(np.all(inc_L2 <= 1e-12)),
        "L2_violations": int(np.sum(inc_L2 > 1e-12)),
        "L2_worst_increase": float(np.max(inc_L2)) if inc_L2.size else 0.0,
    }

# -------------------------
# MDL descent (exact)
# -------------------------

def mdl_descent_exact(mdl: np.ndarray) -> Dict[str, object]:
    inc = mdl[1:] - mdl[:-1]
    return {
        "MDL_monotone": bool(np.all(inc <= 1e-12)),
        "MDL_violations": int(np.sum(inc > 1e-12)),
        "MDL_worst_increase": float(np.max(inc)) if inc.size else 0.0,
        "MDL_worst_iter": int(np.argmax(inc) + 1) if inc.size else -1,
    }

# -------------------------
# 4D closure embedding
# -------------------------

def build_closure_embedding(
    df: pd.DataFrame,
    betas: np.ndarray,
    P: ProjectionDefect,
    iter_col: str = "iter",
    alpha_col: Optional[str] = "alpha",
    mdl_col: Optional[str] = "E_MDL_proxy",
    chi2_col: Optional[str] = "chi2_dof",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    v_t = [ ||Pβ||2 , ||Dβ||2 , depth , arrow ]
      depth = log10(alpha) if available else iter
      arrow = mdl if available else -log(chi2_dof)
    """
    Px = P.P(betas)
    Dx = betas - Px

    v0 = l2(Px, axis=-1)
    v1 = l2(Dx, axis=-1)

    if alpha_col and alpha_col in df.columns:
        depth = np.log10(np.maximum(df[alpha_col].to_numpy(dtype=float), 1e-12))
    else:
        depth = df[iter_col].to_numpy(dtype=float)

    if mdl_col and mdl_col in df.columns:
        arrow = df[mdl_col].to_numpy(dtype=float)
    elif chi2_col and chi2_col in df.columns:
        arrow = -safe_log(df[chi2_col].to_numpy(dtype=float))
    else:
        # fallback: arrow = -||D||2
        arrow = -v1

    V = np.stack([v0, v1, depth, arrow], axis=1)
    aux = {"p_norm": v0, "d_norm": v1, "depth": depth, "arrow": arrow}
    return V, aux

# -------------------------
# Masked quadratic signatures
# -------------------------

def candidate_signatures(dim: int = 4) -> List[Tuple[int, int, int]]:
    """
    Return (p,q,z) with p+q+z=dim and q>=0, etc.
    We'll consider z=0 only by default (nondegenerate), but include z cases too.
    """
    out = []
    for p in range(dim + 1):
        for q in range(dim + 1 - p):
            z = dim - p - q
            out.append((p, q, z))
    # prefer nondegenerate first (z=0)
    out.sort(key=lambda t: (t[2] != 0, -t[0], t[1], t[2]))
    return out

def masked_quadratic(v: np.ndarray, sig: Tuple[int, int, int]) -> float:
    """
    Q(v) = sum_{i< p} v_i^2 - sum_{p <= i < p+q} v_i^2
    remaining z entries ignored
    """
    p, q, z = sig
    vv = v * v
    pos = np.sum(vv[:p])
    neg = np.sum(vv[p:p+q]) if q > 0 else 0.0
    return float(pos - neg)

def signature_fejer_fraction(
    V: np.ndarray,
    sig: Tuple[int, int, int],
    use_abs: bool = False,
) -> Tuple[float, float, int]:
    """
    Fejér-like check on Q along iterations:
        Q(V_{t+1}) <= Q(V_t)
    Optionally use_abs=True to check |Q| descent instead.

    Returns:
      (fraction_satisfied, max_violation, argmax_iter)
    """
    T = V.shape[0]
    Q = np.array([masked_quadratic(V[t], sig) for t in range(T)], dtype=float)
    if use_abs:
        Q = np.abs(Q)
    dQ = Q[1:] - Q[:-1]
    sat = np.sum(dQ <= 1e-12)
    frac = float(sat / max(len(dQ), 1))
    max_viol = float(np.max(dQ)) if dQ.size else 0.0
    argmax = int(np.argmax(dQ) + 1) if dQ.size else -1
    return frac, max_viol, argmax

# -------------------------
# Loading / grouping
# -------------------------

def extract_betas(df: pd.DataFrame, beta_cols: List[str]) -> np.ndarray:
    for c in beta_cols:
        if c not in df.columns:
            raise ValueError(f"Missing beta column: {c}. Available: {list(df.columns)[:30]} ...")
    return df[beta_cols].to_numpy(dtype=float)

def per_label_groups(df: pd.DataFrame, label_col: str = "label") -> Dict[str, pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'. Columns: {df.columns}")
    return {k: g.sort_values("iter") for k, g in df.groupby(label_col)}

# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True, help="CSV with per-step iterations per label.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--iter-col", default="iter")
    ap.add_argument("--alpha-name", default="alpha")
    ap.add_argument("--mdlname", default="E_MDL_proxy")
    ap.add_argument("--chi2name", default="chi2_dof")
    ap.add_argument("--beta-cols", nargs="+", required=True, help="beta columns, e.g. b0 b1 b2 b3 b4")
    ap.add_argument("--P-odd-indices", nargs="+", type=int, default=[1, 3], help="indices to zero for P")
    ap.add_argument("--iters-max", type=int, default=10**9, help="cap iterations by iter <= this")
    ap.add_argument("--y-samples", type=int, default=64, help="how many y in Fix(P) per label")
    ap.add_argument("--noise-sigma", type=float, default=0.05, help="even-only noise sigma for y sampling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mdl-scale", type=float, default=1.0, help="scale for MDL-distance proxy")
    ap.add_argument("--sig-use-abs", action="store_true", help="rank signatures by |Q| descent instead of Q descent")
    args = ap.parse_args()

    ensure_dir(args.out)
    rng = set_seed(args.seed)

    df = pd.read_csv(args.timeseries)
    if args.iter_col not in df.columns:
        raise ValueError(f"Missing iter column '{args.iter_col}' in {args.timeseries}")

    # filter max iter
    df = df[df[args.iter_col].to_numpy(dtype=float) <= args.iters_max].copy()

    groups = per_label_groups(df, label_col=args.label_col)

    P = ProjectionDefect(odd_idx=list(args.P_odd_indices), beta_dim=len(args.beta_cols))
    even_idx = [i for i in range(P.beta_dim) if i not in set(P.odd_idx)]

    rows_defect = []
    rows_mdl = []
    rows_fejer = []
    rows_closest = []
    rows_sig = []
    rows_embed = []

    for label, g in groups.items():
        g = g.sort_values(args.iter_col)
        betas = extract_betas(g, args.beta_cols)

        # Pool of Px for sampling Fix(P)
        Px_pool = P.P(betas)

        y = sample_y_in_fixP(
            rng=rng,
            Px_pool=Px_pool,
            even_idx=even_idx,
            n=args.y_samples,
            noise_sigma=args.noise_sigma,
        )

        # defect monotonicity
        dm = defect_monotonicity(betas, P)
        dm.update({"label": label, "T_iter": int(g[args.iter_col].max()), "beta_dim": P.beta_dim})
        rows_defect.append(dm)

        # mdls
        if args.mdlname in g.columns:
            mdl = g[args.mdlname].to_numpy(dtype=float)
            mm = mdl_descent_exact(mdl)
            mm.update({"label": label, "T_iter": int(g[args.iter_col].max())})
            rows_mdl.append(mm)
        else:
            # fallback: MDL proxy from betas and chi2 if present
            if args.chi2name in g.columns:
                chi2 = g[args.chi2name].to_numpy(dtype=float)
                # crude: MDL = ||Pβ||1 + ||Dβ||1 + log chi2
                Px = P.P(betas)
                Dx = betas - Px
                mdl = l1(Px, axis=-1) + l1(Dx, axis=-1) + safe_log(chi2)
                mm = mdl_descent_exact(mdl)
                mm.update({"label": label, "T_iter": int(g[args.iter_col].max()), "note": "fallback_mdl"})
                rows_mdl.append(mm)

        # Fejér-to-fixed-set
        frac, maxv, argmax = fejer_set_check(betas, P, y, metric="L1", mdl_scale=args.mdl_scale)
        rows_fejer.append({
            "label": label,
            "metric": "L1",
            "fejer_set_frac": frac,
            "fejer_set_max_violation": maxv,
            "fejer_set_argmax": argmax,
            "T_iter": int(g[args.iter_col].max()),
            "beta_dim": P.beta_dim,
        })
        # Optional: MDL-distance Fejér-to-fixed-set
        fracM, maxvM, argmaxM = fejer_set_check(betas, P, y, metric="MDL", mdl_scale=args.mdl_scale)
        rows_fejer.append({
            "label": label,
            "metric": "MDL",
            "fejer_set_frac": fracM,
            "fejer_set_max_violation": maxvM,
            "fejer_set_argmax": argmaxM,
            "T_iter": int(g[args.iter_col].max()),
            "beta_dim": P.beta_dim,
        })

        # ClosestPoint
        cfrac, cmaxv, cargmax = closest_point_check(betas, P, y, metric="L1", mdl_scale=args.mdl_scale)
        rows_closest.append({
            "label": label,
            "metric": "L1",
            "closest_frac": cfrac,
            "closest_max_violation": cmaxv,
            "closest_argmax": cargmax,
            "T_iter": int(g[args.iter_col].max()),
            "beta_dim": P.beta_dim,
        })
        cfracM, cmaxvM, cargmaxM = closest_point_check(betas, P, y, metric="MDL", mdl_scale=args.mdl_scale)
        rows_closest.append({
            "label": label,
            "metric": "MDL",
            "closest_frac": cfracM,
            "closest_max_violation": cmaxvM,
            "closest_argmax": cargmaxM,
            "T_iter": int(g[args.iter_col].max()),
            "beta_dim": P.beta_dim,
        })

        # 4D embedding for signature elimination
        V, aux = build_closure_embedding(
            g, betas, P,
            iter_col=args.iter_col,
            alpha_col=args.alpha_name if args.alpha_name in g.columns else None,
            mdl_col=args.mdlname if args.mdlname in g.columns else None,
            chi2_col=args.chi2name if args.chi2name in g.columns else None,
        )

        # write per-step embedding rows
        iters = g[args.iter_col].to_numpy(dtype=float)
        for t in range(V.shape[0]):
            rows_embed.append({
                "label": label,
                "iter": int(iters[t]),
                "v_pnorm": float(V[t, 0]),
                "v_dnorm": float(V[t, 1]),
                "v_depth": float(V[t, 2]),
                "v_arrow": float(V[t, 3]),
            })

        # rank signatures by Fejér fraction on Q(V_t)
        for sig in candidate_signatures(dim=4):
            fracQ, maxQ, argQ = signature_fejer_fraction(V, sig, use_abs=args.sig_use_abs)
            rows_sig.append({
                "label": label,
                "p": sig[0],
                "q": sig[1],
                "z": sig[2],
                "fejer_frac_Q": fracQ,
                "max_violation_Q": maxQ,
                "argmax_iter_Q": argQ,
                "use_abs": bool(args.sig_use_abs),
            })

    # Save outputs
    pd.DataFrame(rows_defect).to_csv(os.path.join(args.out, "defect_monotonicity_report.csv"), index=False)
    pd.DataFrame(rows_mdl).to_csv(os.path.join(args.out, "mdl_descent_exact.csv"), index=False)
    pd.DataFrame(rows_fejer).to_csv(os.path.join(args.out, "fejer_set_report.csv"), index=False)
    pd.DataFrame(rows_closest).to_csv(os.path.join(args.out, "closestpoint_report.csv"), index=False)
    pd.DataFrame(rows_embed).to_csv(os.path.join(args.out, "closure_embedding_per_step.csv"), index=False)

    sig_df = pd.DataFrame(rows_sig)
    sig_df.to_csv(os.path.join(args.out, "maskedQ_signature_rank_raw.csv"), index=False)

    # Also compute a per-label “best signature” by highest fejer_frac_Q then lowest max_violation_Q
    best_rows = []
    for label, gsig in sig_df.groupby("label"):
        gsig2 = gsig.sort_values(["fejer_frac_Q", "max_violation_Q"], ascending=[False, True])
        best = gsig2.iloc[0].to_dict()
        best_rows.append(best)
    pd.DataFrame(best_rows).to_csv(os.path.join(args.out, "maskedQ_signature_rank.csv"), index=False)

    print("[ok] wrote:", os.path.join(args.out, "defect_monotonicity_report.csv"))
    print("[ok] wrote:", os.path.join(args.out, "mdl_descent_exact.csv"))
    print("[ok] wrote:", os.path.join(args.out, "fejer_set_report.csv"))
    print("[ok] wrote:", os.path.join(args.out, "closestpoint_report.csv"))
    print("[ok] wrote:", os.path.join(args.out, "closure_embedding_per_step.csv"))
    print("[ok] wrote:", os.path.join(args.out, "maskedQ_signature_rank.csv"))
    print("[ok] wrote:", os.path.join(args.out, "maskedQ_signature_rank_raw.csv"))


if __name__ == "__main__":
    main()
