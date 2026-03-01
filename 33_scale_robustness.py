#!/usr/bin/env python3
"""
33_scale_robustness.py

Scale-robustness reporting for a fixed indefinite cone mask.

Reports the pos_scale interval length where cone_frac_min >= threshold
per label and overall, optionally under perturbation variants.

Example:
  python 33_scale_robustness.py \
    --embedding hepdata_lyapunov_test_out_all/dashi_idk_out/closure_embedding_per_step.csv \
    --arrow-col v_depth \
    --x-cols v_pnorm v_dnorm v_arrow \
    --mask -1,1,-1 \
    --pos-scale-min 0.1 --pos-scale-max 1.0 --pos-scale-step 0.05 \
    --thresholds 1.0,0.99 \
    --out-prefix scale_robustness
"""
from __future__ import annotations

import argparse
import math
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LABEL_COL = "label"
ID_COL_CANDIDATES = ["step", "iter", "t", "time", "k"]


def _pick_step_col(df: pd.DataFrame) -> str:
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"no step/iter column found among {ID_COL_CANDIDATES}")


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)


def _default_x_cols(df: pd.DataFrame, arrow_col: str, step_col: str) -> List[str]:
    drop = {LABEL_COL, step_col, arrow_col}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if _is_numeric_series(df[c]):
            cols.append(c)
    return cols


def _parse_list(val: str, cast=float) -> List:
    if val is None:
        return []
    parts = [p.strip() for p in val.split(",") if p.strip() != ""]
    return [cast(p) for p in parts]


def _parse_mask(mask_str: str) -> np.ndarray:
    parts = [p.strip() for p in mask_str.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("mask is empty")
    m = np.array([int(p) for p in parts], dtype=int)
    if not np.all(np.isin(m, [-1, 1])):
        raise ValueError(f"mask must be +/-1 only: {mask_str}")
    return m


def _sig_counts(mask: np.ndarray) -> Tuple[int, int]:
    p = int(np.sum(mask == 1))
    q = int(np.sum(mask == -1))
    return p, q


def _rank_transform(a: np.ndarray) -> np.ndarray:
    # Dense ranks 0..n-1 (stable for ties)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def _arrow_transform(a: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return a
    if name == "rank":
        return _rank_transform(a)
    if name == "log1p_shift":
        shift = -np.min(a) if np.min(a) < 0 else 0.0
        return np.log1p(a + shift)
    if name == "abs":
        return np.abs(a)
    if name == "neg":
        return -a
    raise ValueError(f"unknown arrow transform: {name}")


def _jump_filter_mask(dX: np.ndarray, forward: np.ndarray, frac: float) -> np.ndarray:
    n = len(dX)
    keep = np.ones(n, dtype=bool)
    if frac <= 0:
        return keep
    fidx = np.where(forward)[0]
    if len(fidx) == 0:
        return keep
    norms = np.linalg.norm(dX[fidx], axis=1)
    k_drop = int(math.floor(frac * len(fidx)))
    if k_drop <= 0:
        return keep
    drop_idx = fidx[np.argsort(norms)[-k_drop:]]
    keep[drop_idx] = False
    return keep


def _contiguous_intervals(xs: List[float], ok: List[bool]) -> List[Tuple[float, float]]:
    intervals = []
    start = None
    for i, good in enumerate(ok):
        if good and start is None:
            start = xs[i]
        if (not good or i == len(ok) - 1) and start is not None:
            end = xs[i] if good and i == len(ok) - 1 else xs[i - 1]
            intervals.append((start, end))
            start = None
    return intervals


def _best_interval(xs: List[float], ok: List[bool]) -> Tuple[float, float, float]:
    intervals = _contiguous_intervals(xs, ok)
    if not intervals:
        return float("nan"), float("nan"), 0.0
    best = max(intervals, key=lambda ab: ab[1] - ab[0])
    return best[0], best[1], float(best[1] - best[0])


def _compute_label_stats(
    df: pd.DataFrame,
    x_cols: List[str],
    arrow_col: str,
    arrow_transform: str,
    forward_mode: str,
    step_col: str,
    eps_arrow: float,
    mdl_map: Dict[str, np.ndarray] | None,
    mdl_eps: float,
    mdl_direction: str,
    mdl_quantile: float,
    beta_map: Dict[str, np.ndarray] | None,
    chi2_map: Dict[str, np.ndarray] | None,
    snap_filter: str,
    snap_beta_quantile: float,
    snap_beta_abs: float,
    snap_zero_eps: float,
    snap_zero_min: int,
    snap_chi2_factor: float,
    snap_mdl_eps: float,
    snap_dnorm_quantile: float,
    snap_shrink_kappa: float,
    snap_shrink_min: int,
    noise_std: float,
    noise_scale: str,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, np.ndarray]]:
    # optional noise
    if noise_std > 0:
        if noise_scale == "rel":
            col_std = df[x_cols].std(axis=0).to_numpy(dtype=float)
            scale = np.where(col_std > 0, col_std, 1.0)
            noise = rng.normal(0.0, noise_std, size=df[x_cols].shape) * scale
        else:
            noise = rng.normal(0.0, noise_std, size=df[x_cols].shape)
        df = df.copy()
        df.loc[:, x_cols] = df[x_cols].to_numpy(dtype=float) + noise

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lab, g in df.groupby(LABEL_COL):
        g = g.sort_values(step_col, kind="mergesort")
        X = g[x_cols].to_numpy(dtype=float)
        A_raw = g[arrow_col].to_numpy(dtype=float)
        A = _arrow_transform(A_raw, arrow_transform)
        if len(g) < 2:
            continue
        dX = X[1:] - X[:-1]
        backward = None
        snap_keep = None
        if forward_mode == "order":
            forward = np.ones(len(dX), dtype=bool)
        elif forward_mode == "arrow_rank_freeze":
            r = _rank_transform(A_raw)
            dR = r[1:] - r[:-1]
            forward = dR >= 0
        elif forward_mode == "mdl":
            if mdl_map is None or str(lab) not in mdl_map:
                raise ValueError(f"MDL map missing for label {lab}")
            mdl = mdl_map[str(lab)]
            if len(mdl) != len(g):
                raise ValueError(f"MDL length mismatch for label {lab}: {len(mdl)} vs {len(g)}")
            dM = mdl[1:] - mdl[:-1]
            if mdl_direction == "decrease":
                forward = dM <= (-mdl_eps)
            else:
                forward = dM >= (mdl_eps)
        elif forward_mode == "mdl_quantile":
            if mdl_map is None or str(lab) not in mdl_map:
                raise ValueError(f"MDL map missing for label {lab}")
            mdl = mdl_map[str(lab)]
            if len(mdl) != len(g):
                raise ValueError(f"MDL length mismatch for label {lab}: {len(mdl)} vs {len(g)}")
            dM = mdl[1:] - mdl[:-1]
            q = float(mdl_quantile)
            if not (0.0 < q < 0.5):
                raise ValueError("--mdl-quantile must be in (0, 0.5)")
            lo = np.nanquantile(dM, q)
            hi = np.nanquantile(dM, 1.0 - q)
            forward = dM <= lo
            backward = dM >= hi
        else:
            dA = A[1:] - A[:-1]
            forward = dA >= (-eps_arrow)
        if snap_filter != "none":
            if beta_map is None or str(lab) not in beta_map:
                raise ValueError(f"beta map missing for label {lab}")
            B = beta_map[str(lab)]
            if len(B) != len(g):
                raise ValueError(f"beta length mismatch for label {lab}: {len(B)} vs {len(g)}")
            dB = B[1:] - B[:-1]
            dB_norm = np.linalg.norm(dB, axis=1)
            if snap_filter == "beta_norm_quantile":
                q = float(snap_beta_quantile)
                if not (0.0 < q < 1.0):
                    raise ValueError("--snap-beta-quantile must be in (0,1)")
                thresh = np.nanquantile(dB_norm, q)
                snap_keep = dB_norm <= thresh
            elif snap_filter == "beta_norm_abs":
                snap_keep = dB_norm <= snap_beta_abs
            elif snap_filter == "signature":
                if chi2_map is None or str(lab) not in chi2_map:
                    raise ValueError(f"chi2 map missing for label {lab}")
                chi = chi2_map[str(lab)]
                if len(chi) != len(g):
                    raise ValueError(f"chi2 length mismatch for label {lab}: {len(chi)} vs {len(g)}")
                if mdl_map is None or str(lab) not in mdl_map:
                    raise ValueError(f"MDL map missing for label {lab}")
                mdl = mdl_map[str(lab)]
                if len(mdl) != len(g):
                    raise ValueError(f"MDL length mismatch for label {lab}: {len(mdl)} vs {len(g)}")
                # chi2 spike
                chi_prev = np.maximum(chi[:-1], 1e-12)
                chi_curr = np.maximum(chi[1:], 1e-12)
                chi_ratio = chi_curr / chi_prev
                # mdl descent
                dM = mdl[1:] - mdl[:-1]
                # large beta move (quantile threshold)
                q = float(snap_beta_quantile)
                if not (0.0 < q < 1.0):
                    raise ValueError("--snap-beta-quantile must be in (0,1)")
                thresh = np.nanquantile(dB_norm, q)
                large_beta = dB_norm >= thresh
                # optional zeroing count: components crossing to (near) zero
                if snap_zero_min > 0:
                    prev = B[:-1]
                    curr = B[1:]
                    zeroing = (np.abs(prev) > snap_zero_eps) & (np.abs(curr) <= snap_zero_eps)
                    zero_count = np.sum(zeroing, axis=1)
                    zero_ok = zero_count >= snap_zero_min
                else:
                    zero_ok = np.ones_like(dB_norm, dtype=bool)
                snap = large_beta & (chi_ratio >= snap_chi2_factor) & (dM <= -snap_mdl_eps) & zero_ok
                snap_keep = ~snap
            elif snap_filter == "joint_signature":
                if chi2_map is None or str(lab) not in chi2_map:
                    raise ValueError(f"chi2 map missing for label {lab}")
                chi = chi2_map[str(lab)]
                if len(chi) != len(g):
                    raise ValueError(f"chi2 length mismatch for label {lab}: {len(chi)} vs {len(g)}")
                if mdl_map is None or str(lab) not in mdl_map:
                    raise ValueError(f"MDL map missing for label {lab}")
                mdl = mdl_map[str(lab)]
                if len(mdl) != len(g):
                    raise ValueError(f"MDL length mismatch for label {lab}: {len(mdl)} vs {len(g)}")
                # chi2 spike + mdl descent
                chi_prev = np.maximum(chi[:-1], 1e-12)
                chi_curr = np.maximum(chi[1:], 1e-12)
                chi_ratio = chi_curr / chi_prev
                dM = mdl[1:] - mdl[:-1]
                # large beta move (quantile threshold)
                q = float(snap_beta_quantile)
                if not (0.0 < q < 1.0):
                    raise ValueError("--snap-beta-quantile must be in (0,1)")
                beta_thresh = np.nanquantile(dB_norm, q)
                large_beta = dB_norm >= beta_thresh
                # large |delta dnorm| threshold
                dnorm = np.abs(dX[:, 1])
                qd = float(snap_dnorm_quantile)
                if not (0.0 < qd < 1.0):
                    raise ValueError("--snap-dnorm-quantile must be in (0,1)")
                dnorm_thresh = np.nanquantile(dnorm, qd)
                large_dnorm = dnorm >= dnorm_thresh
                # zeroing count
                if snap_zero_min > 0:
                    prev = B[:-1]
                    curr = B[1:]
                    zeroing = (np.abs(prev) > snap_zero_eps) & (np.abs(curr) <= snap_zero_eps)
                    zero_count = np.sum(zeroing, axis=1)
                    zero_ok = zero_count >= snap_zero_min
                else:
                    zero_ok = np.ones_like(dB_norm, dtype=bool)
                snap = large_beta & large_dnorm & (chi_ratio >= snap_chi2_factor) & (dM <= -snap_mdl_eps) & zero_ok
                snap_keep = ~snap
            elif snap_filter == "shrink_ratio":
                if chi2_map is None or str(lab) not in chi2_map:
                    raise ValueError(f"chi2 map missing for label {lab}")
                chi = chi2_map[str(lab)]
                if len(chi) != len(g):
                    raise ValueError(f"chi2 length mismatch for label {lab}: {len(chi)} vs {len(g)}")
                if mdl_map is None or str(lab) not in mdl_map:
                    raise ValueError(f"MDL map missing for label {lab}")
                mdl = mdl_map[str(lab)]
                if len(mdl) != len(g):
                    raise ValueError(f"MDL length mismatch for label {lab}: {len(mdl)} vs {len(g)}")
                chi_prev = np.maximum(chi[:-1], 1e-12)
                chi_curr = np.maximum(chi[1:], 1e-12)
                chi_ratio = chi_curr / chi_prev
                dM = mdl[1:] - mdl[:-1]
                prev = B[:-1]
                curr = B[1:]
                denom = np.maximum(np.abs(prev), 1e-12)
                shrink = (np.abs(curr) / denom) <= snap_shrink_kappa
                shrink_count = np.sum(shrink, axis=1)
                shrink_ok = shrink_count >= snap_shrink_min
                snap = shrink_ok & (chi_ratio >= snap_chi2_factor) & (dM <= -snap_mdl_eps)
                snap_keep = ~snap
            elif snap_filter == "shrink_dnorm":
                if chi2_map is None or str(lab) not in chi2_map:
                    raise ValueError(f"chi2 map missing for label {lab}")
                chi = chi2_map[str(lab)]
                if len(chi) != len(g):
                    raise ValueError(f"chi2 length mismatch for label {lab}: {len(chi)} vs {len(g)}")
                if mdl_map is None or str(lab) not in mdl_map:
                    raise ValueError(f"MDL map missing for label {lab}")
                mdl = mdl_map[str(lab)]
                if len(mdl) != len(g):
                    raise ValueError(f"MDL length mismatch for label {lab}: {len(mdl)} vs {len(g)}")
                chi_prev = np.maximum(chi[:-1], 1e-12)
                chi_curr = np.maximum(chi[1:], 1e-12)
                chi_ratio = chi_curr / chi_prev
                dM = mdl[1:] - mdl[:-1]
                prev = B[:-1]
                curr = B[1:]
                denom = np.maximum(np.abs(prev), 1e-12)
                shrink = (np.abs(curr) / denom) <= snap_shrink_kappa
                shrink_count = np.sum(shrink, axis=1)
                shrink_ok = shrink_count >= snap_shrink_min
                # dnorm quantile
                dnorm = np.abs(dX[:, 1])
                qd = float(snap_dnorm_quantile)
                if not (0.0 < qd < 1.0):
                    raise ValueError("--snap-dnorm-quantile must be in (0,1)")
                dnorm_thresh = np.nanquantile(dnorm, qd)
                large_dnorm = dnorm >= dnorm_thresh
                snap = shrink_ok & large_dnorm & (chi_ratio >= snap_chi2_factor) & (dM <= -snap_mdl_eps)
                snap_keep = ~snap
            elif snap_filter == "shrink_or_chi2":
                if chi2_map is None or str(lab) not in chi2_map:
                    raise ValueError(f"chi2 map missing for label {lab}")
                chi = chi2_map[str(lab)]
                if len(chi) != len(g):
                    raise ValueError(f"chi2 length mismatch for label {lab}: {len(chi)} vs {len(g)}")
                if mdl_map is None or str(lab) not in mdl_map:
                    raise ValueError(f"MDL map missing for label {lab}")
                mdl = mdl_map[str(lab)]
                if len(mdl) != len(g):
                    raise ValueError(f"MDL length mismatch for label {lab}: {len(mdl)} vs {len(g)}")
                chi_prev = np.maximum(chi[:-1], 1e-12)
                chi_curr = np.maximum(chi[1:], 1e-12)
                chi_ratio = chi_curr / chi_prev
                dM = mdl[1:] - mdl[:-1]
                prev = B[:-1]
                curr = B[1:]
                denom = np.maximum(np.abs(prev), 1e-12)
                shrink = (np.abs(curr) / denom) <= snap_shrink_kappa
                shrink_count = np.sum(shrink, axis=1)
                shrink_ok = shrink_count >= snap_shrink_min
                # dnorm quantile
                dnorm = np.abs(dX[:, 1])
                qd = float(snap_dnorm_quantile)
                if not (0.0 < qd < 1.0):
                    raise ValueError("--snap-dnorm-quantile must be in (0,1)")
                dnorm_thresh = np.nanquantile(dnorm, qd)
                large_dnorm = dnorm >= dnorm_thresh
                snap = shrink_ok & (dM <= -snap_mdl_eps) & (large_dnorm | (chi_ratio >= snap_chi2_factor))
                snap_keep = ~snap
            else:
                raise ValueError(f"unknown snap_filter: {snap_filter}")
        out[str(lab)] = {
            "dX": dX,
            "forward": forward,
            "backward": backward,
            "snap_keep": snap_keep,
        }
    return out


def _sweep_for_variant(
    label_data: Dict[str, Dict[str, np.ndarray]],
    mask: np.ndarray,
    eps: float,
    jump_filter_frac: float,
    pos_scales: List[float],
    resample_steps: bool,
    rng: np.random.Generator,
    two_sided: str,
    min_steps_per_label: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_label_rows = []
    overall_rows = []

    # precompute per-label data
    per_label_data = {}
    for lab, d in label_data.items():
        dX = d["dX"]
        forward = d["forward"]
        backward = d.get("backward")
        keep = _jump_filter_mask(dX, forward, jump_filter_frac)
        snap_keep = d.get("snap_keep")
        if snap_keep is not None:
            keep = keep & snap_keep
        fwd = forward & keep
        if backward is None:
            bwd = (~forward) & keep
        else:
            bwd = backward & keep
        if not np.any(fwd):
            per_label_data[lab] = None
            continue
        idx_f = np.where(fwd)[0]
        if resample_steps:
            idx_f = rng.choice(idx_f, size=len(idx_f), replace=True)
        if two_sided in ("complement", "bidirectional"):
            idx_b = np.where(bwd)[0]
            if resample_steps and len(idx_b) > 0:
                idx_b = rng.choice(idx_b, size=len(idx_b), replace=True)
        else:
            idx_b = np.array([], dtype=int)
        if min_steps_per_label > 1:
            if len(idx_f) < min_steps_per_label or (two_sided in ("complement", "bidirectional") and len(idx_b) < min_steps_per_label):
                per_label_data[lab] = None
                continue
        per_label_data[lab] = (dX[idx_f], dX[idx_b])

    for pos_scale in pos_scales:
        per_label_fracs = []
        for lab, pair in per_label_data.items():
            if pair is None:
                per_label_fracs.append(np.nan)
                continue
            dX_f, dX_b = pair
            if len(dX_f) == 0:
                per_label_fracs.append(np.nan)
                continue
            sq = dX_f * dX_f
            sp = np.sum(sq[:, mask == 1], axis=1) if np.any(mask == 1) else np.zeros(len(dX_f))
            sn = np.sum(sq[:, mask == -1], axis=1) if np.any(mask == -1) else np.zeros(len(dX_f))
            q_f = pos_scale * sp - sn
            frac_f = float(np.mean(q_f <= eps))
            if two_sided == "complement":
                if len(dX_b) == 0:
                    frac = np.nan
                else:
                    sqb = dX_b * dX_b
                    spb = np.sum(sqb[:, mask == 1], axis=1) if np.any(mask == 1) else np.zeros(len(dX_b))
                    snb = np.sum(sqb[:, mask == -1], axis=1) if np.any(mask == -1) else np.zeros(len(dX_b))
                    q_b = pos_scale * spb - snb
                    frac_b = float(np.mean(q_b >= -eps))
                    frac = min(frac_f, frac_b)
            elif two_sided == "bidirectional":
                if len(dX_b) == 0:
                    frac = np.nan
                else:
                    sqb = dX_b * dX_b
                    spb = np.sum(sqb[:, mask == 1], axis=1) if np.any(mask == 1) else np.zeros(len(dX_b))
                    snb = np.sum(sqb[:, mask == -1], axis=1) if np.any(mask == -1) else np.zeros(len(dX_b))
                    q_b = pos_scale * spb - snb
                    frac_b = float(np.mean(q_b <= eps))
                    frac = min(frac_f, frac_b)
            elif two_sided == "reverse_edges":
                q_b = q_f
                frac_b = float(np.mean(q_b >= -eps))
                frac = min(frac_f, frac_b)
            else:
                frac = frac_f
            per_label_fracs.append(frac)
            per_label_rows.append({
                "label": lab,
                "pos_scale": pos_scale,
                "cone_frac": frac,
            })
        overall_rows.append({
            "pos_scale": pos_scale,
            "cone_frac_min": float(np.nanmin(per_label_fracs)) if per_label_fracs else float("nan"),
            "cone_frac_mean": float(np.nanmean(per_label_fracs)) if per_label_fracs else float("nan"),
        })

    per_label_df = pd.DataFrame(per_label_rows)
    if per_label_df.empty:
        per_label_df = pd.DataFrame(columns=["label", "pos_scale", "cone_frac"])
    overall_df = pd.DataFrame(overall_rows)
    return per_label_df, overall_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--arrow-col", default="v_depth")
    ap.add_argument("--arrow-cols", default="")
    ap.add_argument("--arrow-transform", default="identity",
                    choices=["identity", "rank", "log1p_shift", "abs", "neg"])
    ap.add_argument("--forward-mode", default="arrow",
                    choices=["arrow", "order", "arrow_rank_freeze", "mdl", "mdl_quantile"],
                    help="forward selection: arrow threshold, order-only, or arrow rank freeze")
    ap.add_argument("--x-cols", nargs="*", default=None)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--allow-definite", action="store_true")

    ap.add_argument("--pos-scale-min", type=float, default=0.1)
    ap.add_argument("--pos-scale-max", type=float, default=1.0)
    ap.add_argument("--pos-scale-step", type=float, default=0.05)
    ap.add_argument("--thresholds", default="1.0")

    ap.add_argument("--eps-list", default="1e-12")
    ap.add_argument("--eps-arrow-list", default="1e-12")
    ap.add_argument("--jump-filter-fracs", default="0.0")

    ap.add_argument("--noise-stds", default="0.0")
    ap.add_argument("--noise-scale", choices=["abs", "rel"], default="abs")
    ap.add_argument("--noise-seed", type=int, default=0)

    ap.add_argument("--resample-steps", action="store_true")
    ap.add_argument("--resample-reps", type=int, default=1)
    ap.add_argument("--resample-seed", type=int, default=0)
    ap.add_argument("--two-sided", choices=["none", "complement", "reverse_edges", "bidirectional"], default="none",
                    help="two-sided: complement uses backward steps with Q>=-eps; reverse_edges uses -dX on same edges; bidirectional uses backward steps with same Q<=eps")
    ap.add_argument("--min-steps-per-label", type=int, default=1)

    ap.add_argument("--mdl-timeseries", default="")
    ap.add_argument("--mdl-col", default="E_MDL_proxy")
    ap.add_argument("--mdl-fallback", default="chi2_dof")
    ap.add_argument("--mdl-fallback-mode", choices=["neglog", "neglog1p"], default="neglog1p")
    ap.add_argument("--mdl-use-fallback-only", action="store_true")
    ap.add_argument("--mdl-eps", type=float, default=1e-12)
    ap.add_argument("--mdl-direction", choices=["decrease", "increase"], default="decrease")
    ap.add_argument("--mdl-quantile", type=float, default=0.33)
    ap.add_argument("--snap-filter", choices=["none", "beta_norm_quantile", "beta_norm_abs", "signature", "joint_signature", "shrink_ratio", "shrink_dnorm", "shrink_or_chi2"], default="none")
    ap.add_argument("--snap-beta-quantile", type=float, default=0.95)
    ap.add_argument("--snap-beta-abs", type=float, default=0.0)
    ap.add_argument("--snap-beta-cols", default="b0,b1,b2,b3,b4")
    ap.add_argument("--snap-zero-eps", type=float, default=1e-6)
    ap.add_argument("--snap-zero-min", type=int, default=2)
    ap.add_argument("--snap-chi2-factor", type=float, default=10.0)
    ap.add_argument("--snap-mdl-eps", type=float, default=1e-12)
    ap.add_argument("--snap-dnorm-quantile", type=float, default=0.85)
    ap.add_argument("--snap-shrink-kappa", type=float, default=0.1)
    ap.add_argument("--snap-shrink-min", type=int, default=2)

    ap.add_argument("--out-prefix", default="scale_robustness")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    step_col = _pick_step_col(df)
    arrow_cols = _parse_list(args.arrow_cols, str) if args.arrow_cols else [args.arrow_col]
    x_cols = args.x_cols or _default_x_cols(df, arrow_cols[0], step_col)

    mask = _parse_mask(args.mask)
    p, q = _sig_counts(mask)
    if not args.allow_definite and (p == 0 or q == 0):
        raise ValueError(f"mask must be indefinite (p>0,q>0); got p={p}, q={q}")

    pos_scales = list(np.round(np.arange(args.pos_scale_min, args.pos_scale_max + 1e-12, args.pos_scale_step), 12))
    thresholds = _parse_list(args.thresholds, float)
    eps_list = _parse_list(args.eps_list, float)
    eps_arrow_list = _parse_list(args.eps_arrow_list, float)
    jump_filter_fracs = _parse_list(args.jump_filter_fracs, float)
    noise_stds = _parse_list(args.noise_stds, float)

    mdl_map = None
    beta_map = None
    chi2_map = None
    if args.forward_mode in ("mdl", "mdl_quantile"):
        if not args.mdl_timeseries:
            raise ValueError("--mdl-timeseries is required when --forward-mode mdl")
        ts = pd.read_csv(args.mdl_timeseries)
        if LABEL_COL not in ts.columns:
            raise ValueError("timeseries missing label column")
        if step_col not in ts.columns:
            # try to map 'iter' to step_col
            if "iter" in ts.columns and step_col != "iter":
                ts = ts.rename(columns={"iter": step_col})
            else:
                raise ValueError(f"timeseries missing step column {step_col}")
        mdl_map = {}
        for lab, g in ts.groupby(LABEL_COL):
            g = g.sort_values(step_col, kind="mergesort")
            if (not args.mdl_use_fallback_only) and args.mdl_col in g.columns:
                mdl = g[args.mdl_col].to_numpy(dtype=float)
            elif args.mdl_fallback and args.mdl_fallback in g.columns:
                chi = g[args.mdl_fallback].to_numpy(dtype=float)
                chi = np.maximum(chi, args.mdl_eps)
                if args.mdl_fallback_mode == "neglog":
                    mdl = -np.log(chi)
                else:
                    mdl = -np.log1p(chi)
            else:
                raise ValueError(f"timeseries missing {args.mdl_col} and fallback {args.mdl_fallback} for label {lab}")
            mdl_map[str(lab)] = mdl
        if args.snap_filter != "none":
            beta_cols = [c.strip() for c in args.snap_beta_cols.split(",") if c.strip() != ""]
            beta_cols = [c for c in beta_cols if c in ts.columns]
            if not beta_cols:
                raise ValueError("snap filter requires beta columns (b0..b4) in timeseries")
            beta_map = {}
            chi2_map = {}
            for lab, g in ts.groupby(LABEL_COL):
                g = g.sort_values(step_col, kind="mergesort")
                beta_map[str(lab)] = g[beta_cols].to_numpy(dtype=float)
                if "chi2_dof" in g.columns:
                    chi2_map[str(lab)] = g["chi2_dof"].to_numpy(dtype=float)
    elif args.snap_filter != "none":
        if not args.mdl_timeseries:
            raise ValueError("--mdl-timeseries is required when using snap filter")
        ts = pd.read_csv(args.mdl_timeseries)
        if LABEL_COL not in ts.columns:
            raise ValueError("timeseries missing label column")
        if step_col not in ts.columns:
            if "iter" in ts.columns and step_col != "iter":
                ts = ts.rename(columns={"iter": step_col})
            else:
                raise ValueError(f"timeseries missing step column {step_col}")
        beta_cols = [c.strip() for c in args.snap_beta_cols.split(",") if c.strip() != ""] 
        beta_cols = [c for c in beta_cols if c in ts.columns]
        if not beta_cols:
            raise ValueError("snap filter requires beta columns (b0..b4) in timeseries")
        beta_map = {}
        chi2_map = {}
        for lab, g in ts.groupby(LABEL_COL):
            g = g.sort_values(step_col, kind="mergesort")
            beta_map[str(lab)] = g[beta_cols].to_numpy(dtype=float)
            if "chi2_dof" in g.columns:
                chi2_map[str(lab)] = g["chi2_dof"].to_numpy(dtype=float)

    summary_rows = []
    per_label_interval_rows = []

    variant_id = 0
    for arrow_col, eps, eps_arrow, jf, noise_std in product(
        arrow_cols, eps_list, eps_arrow_list, jump_filter_fracs, noise_stds
    ):
        for rep in range(max(1, args.resample_reps)):
            variant_id += 1
            rng = np.random.default_rng(args.noise_seed + variant_id)
            res_rng = np.random.default_rng(args.resample_seed + variant_id + rep)

            label_data = _compute_label_stats(
                df=df,
                x_cols=x_cols,
                arrow_col=arrow_col,
                arrow_transform=args.arrow_transform,
                forward_mode=args.forward_mode,
                step_col=step_col,
                eps_arrow=eps_arrow,
                mdl_map=mdl_map,
                mdl_eps=args.mdl_eps,
                mdl_direction=args.mdl_direction,
                mdl_quantile=args.mdl_quantile,
                beta_map=beta_map,
                chi2_map=chi2_map,
                snap_filter=args.snap_filter,
                snap_beta_quantile=args.snap_beta_quantile,
                snap_beta_abs=args.snap_beta_abs,
                snap_zero_eps=args.snap_zero_eps,
                snap_zero_min=args.snap_zero_min,
                snap_chi2_factor=args.snap_chi2_factor,
                snap_mdl_eps=args.snap_mdl_eps,
                snap_dnorm_quantile=args.snap_dnorm_quantile,
                snap_shrink_kappa=args.snap_shrink_kappa,
                snap_shrink_min=args.snap_shrink_min,
                noise_std=noise_std,
                noise_scale=args.noise_scale,
                rng=rng,
            )

            per_label_df, overall_df = _sweep_for_variant(
                label_data=label_data,
                mask=mask,
                eps=eps,
                jump_filter_frac=jf,
                pos_scales=pos_scales,
                resample_steps=args.resample_steps,
                rng=res_rng,
                two_sided=args.two_sided,
                min_steps_per_label=args.min_steps_per_label,
            )

            for thr in thresholds:
                ok = overall_df["cone_frac_min"] >= thr
                start, end, length = _best_interval(pos_scales, ok.tolist())
                summary_rows.append({
                    "variant_id": variant_id,
                    "arrow_col": arrow_col,
                    "arrow_transform": args.arrow_transform,
                    "forward_mode": args.forward_mode,
                    "two_sided": args.two_sided,
                    "eps": eps,
                    "eps_arrow": eps_arrow,
                    "jump_filter_frac": jf,
                    "noise_std": noise_std,
                    "noise_scale": args.noise_scale,
                    "resample_steps": args.resample_steps,
                    "resample_rep": rep,
                    "threshold": thr,
                    "overall_interval_start": start,
                    "overall_interval_end": end,
                    "overall_interval_len": length,
                    "n_labels": int(per_label_df["label"].nunique()),
                })

                for lab, g in per_label_df.groupby("label"):
                    ok_l = g["cone_frac"] >= thr
                    s_l, e_l, len_l = _best_interval(pos_scales, ok_l.tolist())
                    per_label_interval_rows.append({
                        "variant_id": variant_id,
                        "label": lab,
                        "arrow_col": arrow_col,
                        "arrow_transform": args.arrow_transform,
                        "forward_mode": args.forward_mode,
                        "two_sided": args.two_sided,
                        "eps": eps,
                        "eps_arrow": eps_arrow,
                        "jump_filter_frac": jf,
                        "noise_std": noise_std,
                        "noise_scale": args.noise_scale,
                        "resample_steps": args.resample_steps,
                        "resample_rep": rep,
                        "threshold": thr,
                        "interval_start": s_l,
                        "interval_end": e_l,
                        "interval_len": len_l,
                    })

    summary_df = pd.DataFrame(summary_rows)
    per_label_interval_df = pd.DataFrame(per_label_interval_rows)

    summary_df.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    per_label_interval_df.to_csv(f"{args.out_prefix}_per_label.csv", index=False)

    print(f"Wrote {args.out_prefix}_summary.csv and {args.out_prefix}_per_label.csv")


if __name__ == "__main__":
    main()
