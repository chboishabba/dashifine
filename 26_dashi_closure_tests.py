#!/usr/bin/env python3
"""
DASHI closure seam-certificate tests for LHC beta timeseries.

Inputs:
  - per_label_timeseries.csv (must include: label, iter, b0..b{d-1})
    Optional: E_MDL_proxy column (if missing, we compute a proxy).

Outputs (CSV):
  - mdl_descent_exact.csv
  - fejer_set_report.csv
  - closestpoint_report.csv
  - transinv_report.csv
  - maskedQ_signature_rank.csv

Core ideas:
  - Define Projection P on beta-space by zeroing a set of "defect" coordinates.
  - Define translation-invariant distances: L1, L2, and an MDL-distance on differences.
  - Fejér-to-fixed-set: d(Px, y) <= d(x, y) for random y in Fix(P).
  - ClosestPoint: d(x, Px) <= d(x, y) for random y in Fix(P).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def stable_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------
# Projection / Defect
# ----------------------------

@dataclass(frozen=True)
class BetaProjection:
    """Projection P that zeros defect indices."""
    dim: int
    defect_idx: Tuple[int, ...]  # indices set to 0 by P

    def P(self, x: np.ndarray) -> np.ndarray:
        y = np.array(x, dtype=float, copy=True)
        y[list(self.defect_idx)] = 0.0
        return y

    def D(self, x: np.ndarray) -> np.ndarray:
        return x - self.P(x)

    def sample_fix_point(self, base: np.ndarray, rng: np.random.Generator,
                         noise_scale: float,
                         support_idx: Tuple[int, ...]) -> np.ndarray:
        """
        Sample y ∈ Fix(P) by taking P(base) and adding noise only on non-defect coords.
        Ensures defect coords remain 0.
        """
        y = self.P(base)
        if noise_scale > 0:
            noise = rng.normal(loc=0.0, scale=noise_scale, size=self.dim)
            mask = np.zeros(self.dim, dtype=bool)
            mask[list(support_idx)] = True
            # Only allow noise on chosen support indices
            y = y + noise * mask.astype(float)
            # Enforce Fix(P)
            y[list(self.defect_idx)] = 0.0
        return y


# ----------------------------
# Distances (translation-invariant)
# ----------------------------

def d_L1(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.abs(x - y)))


def d_L2(x: np.ndarray, y: np.ndarray) -> float:
    v = x - y
    return float(np.sqrt(np.sum(v * v)))


def mdl_proxy_beta(v: np.ndarray,
                   model_idx: Tuple[int, ...],
                   defect_idx: Tuple[int, ...],
                   w_defect: float = 1.0,
                   w_model: float = 1.0) -> float:
    """
    A simple *translation-invariant* MDL-like proxy on a difference vector v.
      MDL(v) = w_model * ||v_model||_1 + w_defect * ||v_defect||_1

    This is the safest "metric-like" surrogate you can use for Fejér/ClosestPoint checks.
    """
    vm = v[list(model_idx)] if model_idx else np.array([], dtype=float)
    vd = v[list(defect_idx)] if defect_idx else np.array([], dtype=float)
    return float(w_model * np.sum(np.abs(vm)) + w_defect * np.sum(np.abs(vd)))


def d_MDL(x: np.ndarray, y: np.ndarray,
          model_idx: Tuple[int, ...],
          defect_idx: Tuple[int, ...],
          w_defect: float,
          w_model: float) -> float:
    return mdl_proxy_beta(x - y, model_idx, defect_idx, w_defect=w_defect, w_model=w_model)


def maskedQ(v: np.ndarray, signs: np.ndarray) -> float:
    """
    Q(v) = sum_i s_i * v_i^2 with s_i ∈ {+1, -1, 0}
    """
    return float(np.sum(signs * (v * v)))


def d_maskedQ(x: np.ndarray, y: np.ndarray, signs: np.ndarray) -> float:
    """
    Distance surrogate from masked quadratic:
      d_Q(x,y) = sqrt(|Q(x-y)|)
    """
    q = maskedQ(x - y, signs)
    return float(np.sqrt(abs(q)))


# ----------------------------
# Loading betas
# ----------------------------

def load_timeseries(path: str, dim: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["label", "iter"] + [f"b{i}" for i in range(dim)]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")
    df = df.sort_values(["label", "iter"]).reset_index(drop=True)
    return df


def get_beta_matrix(df_label: pd.DataFrame, dim: int) -> np.ndarray:
    return df_label[[f"b{i}" for i in range(dim)]].to_numpy(dtype=float)


# ----------------------------
# Tests
# ----------------------------

def test_mdl_descent_exact(df: pd.DataFrame,
                          dim: int,
                          proj: BetaProjection,
                          eps: float,
                          use_existing_E: bool,
                          w_defect: float,
                          w_model: float) -> pd.DataFrame:
    """
    Exact monotone check: MDL(x_{t+1}) <= MDL(x_t) + eps for all steps.
    If E_MDL_proxy exists and use_existing_E=True, use it; else compute
    translation-invariant proxy from betas:
      E(t) = MDL_proxy_beta(beta_t, model_idx, defect_idx)
    """
    rows = []
    model_idx = tuple(i for i in range(dim) if i not in proj.defect_idx)

    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        betas = get_beta_matrix(g, dim)
        iters = g["iter"].to_numpy()

        if use_existing_E and "E_MDL_proxy" in g.columns:
            E = g["E_MDL_proxy"].to_numpy(dtype=float)
        else:
            E = np.array([mdl_proxy_beta(b, model_idx, proj.defect_idx,
                                         w_defect=w_defect, w_model=w_model)
                          for b in betas], dtype=float)

        diffs = E[1:] - E[:-1]
        violations = np.where(diffs > eps)[0]
        ok_all = (len(violations) == 0)
        worst_inc = float(diffs.max()) if len(diffs) else 0.0
        worst_i = int(violations[0] + 1) if len(violations) else -1

        rows.append({
            "label": label,
            "last_iter": int(iters.max()),
            "n_steps": int(len(E) - 1),
            "eps": float(eps),
            "mdl_monotone_all": bool(ok_all),
            "n_violations": int(len(violations)),
            "worst_increase": float(worst_inc),
            "first_violation_iter": int(iters[worst_i]) if worst_i >= 0 else -1
        })

    return pd.DataFrame(rows)


def test_translation_invariance(df: pd.DataFrame,
                                dim: int,
                                dist_name: str,
                                dist_fn: Callable[[np.ndarray, np.ndarray], float],
                                n_trials: int,
                                seed: int) -> pd.DataFrame:
    """
    Numeric TI check:
      d(x+z, y+z) == d(x,y)
    on random points drawn from observed beta support.
    """
    rng = stable_rng(seed)
    rows = []

    for label, g in df.groupby("label", sort=False):
        betas = get_beta_matrix(g, dim)
        if len(betas) < 2:
            continue

        worst = 0.0
        ok = 0
        for _ in range(n_trials):
            i, j, k = rng.integers(0, len(betas), size=3)
            x, y, z = betas[i], betas[j], betas[k]
            a = dist_fn(x + z, y + z)
            b = dist_fn(x, y)
            err = abs(a - b)
            worst = max(worst, err)
            if err <= 1e-12:
                ok += 1

        rows.append({
            "label": label,
            "metric": dist_name,
            "transinv_frac": ok / n_trials if n_trials else 1.0,
            "worst_abs_err": worst,
            "n_trials": n_trials
        })

    return pd.DataFrame(rows)


def test_fejer_set(df: pd.DataFrame,
                   dim: int,
                   proj: BetaProjection,
                   metric_name: str,
                   dist: Callable[[np.ndarray, np.ndarray], float],
                   n_y: int,
                   noise_scale: float,
                   seed: int) -> pd.DataFrame:
    """
    Fejér-to-fixed-set:
      d(Px_t, y) <= d(x_t, y)  for random y ∈ Fix(P)
    We generate y from a random base point's P(base) + noise on model coords only.

    Returns per-label:
      fejer_set_all, fejer_set_frac, worst_violation, worst_y_idx, worst_t_idx
    """
    rng = stable_rng(seed)
    rows = []
    model_idx = tuple(i for i in range(dim) if i not in proj.defect_idx)

    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        betas = get_beta_matrix(g, dim)
        if len(betas) == 0:
            continue

        # Precompute Px_t
        Pbetas = np.stack([proj.P(b) for b in betas], axis=0)

        # sample y's in Fix(P)
        ys = []
        for _ in range(n_y):
            base = betas[int(rng.integers(0, len(betas)))]
            ys.append(proj.sample_fix_point(base, rng, noise_scale, support_idx=model_idx))
        ys = np.stack(ys, axis=0)

        total = len(betas) * n_y
        ok = 0
        worst = -1e9
        worst_y = -1
        worst_t = -1

        for t in range(len(betas)):
            x = betas[t]
            px = Pbetas[t]
            for yi in range(n_y):
                y = ys[yi]
                lhs = dist(px, y)
                rhs = dist(x, y)
                gap = lhs - rhs
                if gap <= 0:
                    ok += 1
                if gap > worst:
                    worst = gap
                    worst_y = yi
                    worst_t = t

        rows.append({
            "label": label,
            "metric": metric_name,
            "fejer_set_all": int(ok == total),
            "fejer_set_frac": ok / total if total else 1.0,
            "worst_violation": float(worst),
            "worst_y_idx": int(worst_y),
            "worst_t_idx": int(worst_t),
            "n_y": int(n_y),
            "T_iter": int(g["iter"].max())
        })

    return pd.DataFrame(rows)


def test_closest_point(df: pd.DataFrame,
                       dim: int,
                       proj: BetaProjection,
                       metric_name: str,
                       dist: Callable[[np.ndarray, np.ndarray], float],
                       n_y: int,
                       noise_scale: float,
                       seed: int) -> pd.DataFrame:
    """
    ClosestPoint:
      d(x_t, P x_t) <= d(x_t, y) for random y ∈ Fix(P)

    Returns per-label:
      closest_all, closest_frac, worst_margin (negative means safe, positive violates),
      worst_t, worst_y, worst_iter
    """
    rng = stable_rng(seed)
    rows = []
    model_idx = tuple(i for i in range(dim) if i not in proj.defect_idx)

    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        betas = get_beta_matrix(g, dim)
        iters = g["iter"].to_numpy()
        if len(betas) == 0:
            continue

        # generate y's in Fix(P)
        ys = []
        for _ in range(n_y):
            base = betas[int(rng.integers(0, len(betas)))]
            ys.append(proj.sample_fix_point(base, rng, noise_scale, support_idx=model_idx))
        ys = np.stack(ys, axis=0)

        total = len(betas) * n_y
        ok = 0
        worst_margin = -1e9
        worst_t = -1
        worst_y = -1
        worst_iter = -1

        for t in range(len(betas)):
            x = betas[t]
            px = proj.P(x)
            dxpx = dist(x, px)
            for yi in range(n_y):
                y = ys[yi]
                dxy = dist(x, y)
                margin = dxpx - dxy
                if margin <= 0:
                    ok += 1
                if margin > worst_margin:
                    worst_margin = margin
                    worst_t = t
                    worst_y = yi
                    worst_iter = int(iters[t])

        rows.append({
            "label": label,
            "metric": metric_name,
            "closest_frac": ok / total if total else 1.0,
            "closest_all": int(ok == total),
            "worst_margin": float(worst_margin),
            "worst_t": int(worst_t),
            "worst_y": int(worst_y),
            "worst_iter": int(worst_iter),
            "n_pairs": int(total)
        })

    return pd.DataFrame(rows)


def rank_signatures_maskedQ(df: pd.DataFrame,
                            dim: int,
                            proj: BetaProjection,
                            candidates: List[Tuple[str, np.ndarray]],
                            n_y: int,
                            noise_scale: float,
                            seed: int) -> pd.DataFrame:
    """
    Try candidate masked quadratic signatures, using Fejér-to-fixed-set fraction
    under d_Q(x,y)=sqrt(|Q(x-y)|) as a score.

    This is not "proof", it's the seam-certificate that tells you which signatures
    are compatible with your observed invariant cone geometry.
    """
    rng = stable_rng(seed)
    rows = []
    model_idx = tuple(i for i in range(dim) if i not in proj.defect_idx)

    # Pre-sample y's per label for fairness across candidates
    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        betas = get_beta_matrix(g, dim)
        if len(betas) == 0:
            continue
        Pbetas = np.stack([proj.P(b) for b in betas], axis=0)

        ys = []
        for _ in range(n_y):
            base = betas[int(rng.integers(0, len(betas)))]
            ys.append(proj.sample_fix_point(base, rng, noise_scale, support_idx=model_idx))
        ys = np.stack(ys, axis=0)

        for name, signs in candidates:
            def distQ(a: np.ndarray, b: np.ndarray) -> float:
                return d_maskedQ(a, b, signs)

            total = len(betas) * n_y
            ok = 0
            worst = -1e9

            for t in range(len(betas)):
                x = betas[t]
                px = Pbetas[t]
                for yi in range(n_y):
                    y = ys[yi]
                    gap = distQ(px, y) - distQ(x, y)
                    if gap <= 0:
                        ok += 1
                    worst = max(worst, gap)

            rows.append({
                "label": label,
                "signature_name": name,
                "fejer_set_frac": ok / total if total else 1.0,
                "worst_violation": float(worst),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        # aggregate across labels: mean score, max worst_violation
        agg = out.groupby("signature_name", as_index=False).agg(
            fejer_mean=("fejer_set_frac", "mean"),
            fejer_min=("fejer_set_frac", "min"),
            worst_max=("worst_violation", "max"),
            n_labels=("label", "nunique"),
        ).sort_values(["fejer_mean", "fejer_min"], ascending=False)
        return out.merge(agg, on="signature_name", how="left")
    return out


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True,
                    help="Path to per_label_timeseries.csv")
    ap.add_argument("--out", required=True,
                    help="Output directory")
    ap.add_argument("--dim", type=int, default=5,
                    help="Beta dimension (default: 5 for b0..b4)")
    ap.add_argument("--defect-idx", default="1,3",
                    help="Comma list of defect indices zeroed by P (default: 1,3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-y", type=int, default=512,
                    help="How many random y in Fix(P) per label (default: 512)")
    ap.add_argument("--noise-scale", type=float, default=0.5,
                    help="Noise scale for sampling y in Fix(P) (default: 0.5)")
    ap.add_argument("--ti-trials", type=int, default=5000,
                    help="Translation invariance trials per label (default: 5000)")
    ap.add_argument("--mdl-eps", type=float, default=0.0,
                    help="Epsilon tolerance for exact MDL descent (default: 0)")
    ap.add_argument("--use-existing-E", action="store_true",
                    help="Use E_MDL_proxy column if present; otherwise compute MDL proxy from betas")
    ap.add_argument("--w-model", type=float, default=1.0,
                    help="MDL proxy weight on model coords (default: 1.0)")
    ap.add_argument("--w-defect", type=float, default=1.0,
                    help="MDL proxy weight on defect coords (default: 1.0)")
    args = ap.parse_args()

    ensure_dir(args.out)

    defect_idx = tuple(parse_int_list(args.defect_idx))
    proj = BetaProjection(dim=args.dim, defect_idx=defect_idx)

    df = load_timeseries(args.timeseries, args.dim)

    # Distances
    model_idx = tuple(i for i in range(args.dim) if i not in defect_idx)
    dist_mdl = lambda a, b: d_MDL(a, b, model_idx, defect_idx, args.w_defect, args.w_model)

    # 1) Exact MDL descent
    mdl_report = test_mdl_descent_exact(
        df, args.dim, proj, eps=args.mdl_eps,
        use_existing_E=args.use_existing_E,
        w_defect=args.w_defect, w_model=args.w_model
    )
    mdl_path = os.path.join(args.out, "mdl_descent_exact.csv")
    mdl_report.to_csv(mdl_path, index=False)

    # 2) Translation invariance
    ti_L1 = test_translation_invariance(df, args.dim, "L1", d_L1, args.ti_trials, args.seed)
    ti_L2 = test_translation_invariance(df, args.dim, "L2", d_L2, args.ti_trials, args.seed)
    ti_MDL = test_translation_invariance(df, args.dim, "MDL", dist_mdl, args.ti_trials, args.seed)
    ti = pd.concat([ti_L1, ti_L2, ti_MDL], ignore_index=True)
    ti_path = os.path.join(args.out, "transinv_report.csv")
    ti.to_csv(ti_path, index=False)

    # 3) Fejér-to-fixed-set
    fejer_L1 = test_fejer_set(df, args.dim, proj, "L1", d_L1, args.n_y, args.noise_scale, args.seed)
    fejer_MDL = test_fejer_set(df, args.dim, proj, "MDL", dist_mdl, args.n_y, args.noise_scale, args.seed)
    fejer = pd.concat([fejer_L1, fejer_MDL], ignore_index=True)
    fejer_path = os.path.join(args.out, "fejer_set_report.csv")
    fejer.to_csv(fejer_path, index=False)

    # 4) Closest point
    cp_L1 = test_closest_point(df, args.dim, proj, "L1", d_L1, args.n_y, args.noise_scale, args.seed)
    cp_MDL = test_closest_point(df, args.dim, proj, "MDL", dist_mdl, args.n_y, args.noise_scale, args.seed)
    cp = pd.concat([cp_L1, cp_MDL], ignore_index=True)
    cp_path = os.path.join(args.out, "closestpoint_report.csv")
    cp.to_csv(cp_path, index=False)

    # 5) MaskedQuadratic signatures (you can expand these)
    # Note: dimension here is beta-dim; for a true (3,1) claim, you’ll typically
    # map into the 4D closure space first. This gives you an empirical elimination
    # signal already.
    def sig(name: str, signs_list: List[int]) -> Tuple[str, np.ndarray]:
        arr = np.array(signs_list, dtype=float)
        if len(arr) != args.dim:
            raise ValueError(f"Signature '{name}' has len {len(arr)} but dim is {args.dim}")
        return (name, arr)

    # Some reasonable candidates in 5D beta-space:
    candidates = [
        sig("all_pos_(5,0)", [+1, +1, +1, +1, +1]),
        sig("one_neg_(4,1)_b4", [+1, +1, +1, +1, -1]),
        sig("one_neg_(4,1)_b0", [-1, +1, +1, +1, +1]),
        sig("two_neg_(3,2)",   [+1, +1, +1, -1, -1]),
        sig("split_(2,2)_+0",  [+1, +1, -1, -1,  0]),
        sig("three_neg_(2,3)", [+1, +1, -1, -1, -1]),
    ]

    mq_rank = rank_signatures_maskedQ(df, args.dim, proj, candidates,
                                      n_y=args.n_y, noise_scale=args.noise_scale, seed=args.seed)
    mq_path = os.path.join(args.out, "maskedQ_signature_rank.csv")
    mq_rank.to_csv(mq_path, index=False)

    print("[ok] wrote:", mdl_path)
    print("[ok] wrote:", ti_path)
    print("[ok] wrote:", fejer_path)
    print("[ok] wrote:", cp_path)
    print("[ok] wrote:", mq_path)
    print()
    print("Notes:")
    print("- If you want *exact* MDL monotone descent, run with --mdl-eps 0 and (optionally) tune --w-defect/--w-model.")
    print("- Fejér-to-fixed-set depends strongly on how you sample y ∈ Fix(P). Noise only on model coords is the cleanest seam-certificate.")
    print("- For signature (3,1) you typically want to test in the 4D closure space; this script ranks signatures in your observed beta-space as a first elimination pass.")


if __name__ == "__main__":
    main()
