#!/usr/bin/env python3
"""
31_try_both_delta_cone.py

One script that tries BOTH:
  (A) jump-step filtering (remove top-k% largest ||Δx|| forward steps, per label)
  (B) weighted Lorentz cone via a single positive-axis scale (pos_scale)
      Q(Δx) = pos_scale * sum_{i:sign=+1} (Δx_i^2) - sum_{i:sign=-1} (Δx_i^2)

It screens signatures on Δx only (no Q(x) descent), and uses an "arrow" column ONLY to
define forward steps (Δarrow >= -eps_arrow). Arrow is excluded from Q.

Outputs:
  - delta_cone_signature_rank.csv (combined ranking across methods)
  - delta_cone_violations_<method>.csv (optional, for best signature per method)

Typical usage:
  python 31_try_both_delta_cone.py \
    --embedding path/to/closure_embedding_per_step.csv \
    --arrow-col v_depth \
    --x-cols v_pnorm v_dnorm v_arrow \
    --require-nondegenerate \
    --require-indefinite \
    --eps 1e-12 --eps-arrow 1e-12 \
    --jump-filter-frac 0.02 \
    --dump-violations

Notes:
  * If you don't pass --x-cols, script uses all numeric columns except label/iter/step and arrow col.
  * "Weighted" method chooses pos_scale = min(1.0, min_r) where r = sn/sp over forward steps with sp>0.
    This is the *smallest* change from pos_scale=1 that makes ALL such steps satisfy Q<=0.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def compute_delta_scales(dX: np.ndarray, method: str, eps: float) -> np.ndarray:
    """Compute per-axis scale for Δx normalization."""
    dX = np.asarray(dX, float)
    if dX.ndim != 2:
        raise ValueError("dX must be 2D")
    D = dX.shape[1]
    scales = np.zeros(D, float)
    for j in range(D):
        col = dX[:, j]
        if method == "std":
            s = float(np.nanstd(col))
        elif method == "mad":
            s = _mad(col)
        else:
            s = 1.0
        if (not np.isfinite(s)) or (s < eps):
            s = eps
        scales[j] = s
    return scales

def apply_delta_norm(dX: np.ndarray, scales: np.ndarray) -> np.ndarray:
    dX = np.asarray(dX, float)
    scales = np.asarray(scales, float)
    return dX / scales[None, :]

import pandas as pd


ID_COL_CANDIDATES = ["step", "iter", "t", "time", "k"]
LABEL_COL = "label"


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


def _all_sign_masks(d: int) -> List[np.ndarray]:
    # all ±1 masks
    masks = []
    for bits in range(1 << d):
        m = np.empty(d, dtype=int)
        for i in range(d):
            m[i] = 1 if (bits >> i) & 1 else -1
        masks.append(m)
    return masks


def _sig_counts(mask: np.ndarray) -> Tuple[int, int, int]:
    p = int(np.sum(mask == 1))
    q = int(np.sum(mask == -1))
    z = 0
    return p, q, z


@dataclass
class MethodResult:
    method: str
    p: int
    q: int
    z: int
    mask: str
    pos_scale: float
    forward_frac_min: float
    forward_frac_mean: float
    cone_frac_min: float
    cone_frac_mean: float
    max_Qd_violation_max: float
    n_labels: int
    n_steps_total: int


def compute_deltas(
    df: pd.DataFrame,
    x_cols: List[str],
    arrow_col: str,
    step_col: str,
    eps_arrow: float,
    delta_norm: str,
    delta_norm_eps: float,
    delta_norm_scope: str,
    global_scales: Optional[np.ndarray],
    whiten_mat: Optional[np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns dict[label] with:
      dX: (n_steps, d)
      sp2: per-step sum squares of positive axes (computed later per mask)
      arrow_fwd: bool (n_steps,)
      step_from: step indices for the *from* row
      step_to: step indices for the *to* row
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lab, g in df.groupby(LABEL_COL):
        g = g.sort_values(step_col, kind="mergesort")
        X = g[x_cols].to_numpy(dtype=float)
        A = g[arrow_col].to_numpy(dtype=float)

        if len(g) < 2:
            continue

        dX = X[1:] - X[:-1]
        dA = A[1:] - A[:-1]
        forward = dA >= (-eps_arrow)

        # Optional Δx normalization (per-axis).
        if delta_norm != "none":
            if delta_norm_scope == "global" and global_scales is not None:
                dX = apply_delta_norm(dX, global_scales)
            else:
                dX_f = dX[forward]
                if dX_f.size == 0:
                    scales = np.ones(dX.shape[1], float)
                else:
                    scales = compute_delta_scales(dX_f, delta_norm, delta_norm_eps)
                dX = apply_delta_norm(dX, scales)
        if whiten_mat is not None:
            dX = dX @ whiten_mat.T

        out[str(lab)] = {
            "dX": dX,
            "dA": dA,
            "forward": forward,
            "step_from": g[step_col].to_numpy()[:-1],
            "step_to": g[step_col].to_numpy()[1:],
        }
    return out


def jump_filter_mask_per_label(dX: np.ndarray, forward: np.ndarray, frac: float) -> np.ndarray:
    """
    Returns keep mask for steps (same length as dX), removing largest ||dX|| among forward steps.
    frac in [0,1): fraction of forward steps to drop.
    """
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
    # drop largest k_drop
    drop_idx = fidx[np.argsort(norms)[-k_drop:]]
    keep[drop_idx] = False
    return keep


def eval_mask_for_label(
    dX: np.ndarray,
    forward: np.ndarray,
    keep: np.ndarray,
    mask: np.ndarray,
    eps: float,
    pos_scale: float,
) -> Tuple[float, float, float]:
    """
    Evaluate for one label.
    Returns (forward_frac, cone_forward_frac, max_violation)
    where violation = max(Qd - eps) over forward&keep steps.
    """
    n = len(dX)
    if n == 0:
        return 0.0, 0.0, float("nan")

    fwd = forward & keep
    forward_frac = float(np.mean(forward)) if n > 0 else 0.0

    if not np.any(fwd):
        # no forward steps: define cone frac as NaN; caller handles
        return forward_frac, float("nan"), float("nan")

    # squares
    sq = dX * dX
    sp = np.sum(sq[:, mask == 1], axis=1) if np.any(mask == 1) else np.zeros(n)
    sn = np.sum(sq[:, mask == -1], axis=1) if np.any(mask == -1) else np.zeros(n)

    Qd = pos_scale * sp - sn
    cone_ok = Qd <= eps
    cone_forward_frac = float(np.mean(cone_ok[fwd]))

    viol = Qd - eps
    max_viol = float(np.max(viol[fwd])) if np.any(fwd) else float("nan")
    return forward_frac, cone_forward_frac, max_viol


def pick_pos_scale_minfix(
    dX_all: np.ndarray,
    forward_all: np.ndarray,
    keep_all: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Choose pos_scale to be the smallest downward adjustment from 1.0 that makes all forward&keep steps satisfy Q<=0
    (with eps assumed ~0). Returns min(1.0, min_r) where r = sn/sp for sp>0. If no sp>0 steps, returns 1.0.
    """
    fwd = forward_all & keep_all
    if not np.any(fwd):
        return 1.0
    sq = dX_all * dX_all
    sp = np.sum(sq[:, mask == 1], axis=1) if np.any(mask == 1) else np.zeros(len(dX_all))
    sn = np.sum(sq[:, mask == -1], axis=1) if np.any(mask == -1) else np.zeros(len(dX_all))

    sp_f = sp[fwd]
    sn_f = sn[fwd]
    # Only consider steps where positive part is nonzero; otherwise Q = -sn <=0 already.
    nz = sp_f > 0
    if not np.any(nz):
        return 1.0
    r = sn_f[nz] / sp_f[nz]
    min_r = float(np.min(r))
    return min(1.0, min_r)


def flatten_all(label_data: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dXs, fwds, keeps = [], [], []
    for lab, d in label_data.items():
        dX = d["dX"]
        fwd = d["forward"]
        keep = d["keep"]
        dXs.append(dX)
        fwds.append(fwd)
        keeps.append(keep)
    return np.vstack(dXs), np.concatenate(fwds), np.concatenate(keeps)


def dump_violations_csv(
    path: str,
    df: pd.DataFrame,
    x_cols: List[str],
    arrow_col: str,
    step_col: str,
    mask: np.ndarray,
    pos_scale: float,
    eps: float,
    eps_arrow: float,
    jump_filter_frac: float,
    top_k: int,
    delta_norm: str,
    delta_norm_eps: float,
    delta_norm_scope: str,
    global_scales: Optional[np.ndarray],
    whiten_mat: Optional[np.ndarray],
) -> None:
    deltas = compute_deltas(
        df, x_cols, arrow_col, step_col, eps_arrow,
        delta_norm, delta_norm_eps, delta_norm_scope, global_scales, whiten_mat
    )
    rows = []
    for lab, d in deltas.items():
        dX = d["dX"]; dA = d["dA"]; fwd = d["forward"]
        keep = jump_filter_mask_per_label(dX, fwd, jump_filter_frac)
        sq = dX * dX
        sp = np.sum(sq[:, mask == 1], axis=1) if np.any(mask == 1) else np.zeros(len(dX))
        sn = np.sum(sq[:, mask == -1], axis=1) if np.any(mask == -1) else np.zeros(len(dX))
        Qd = pos_scale * sp - sn
        viol = Qd - eps
        idx = np.where((fwd & keep) & (viol > 0))[0]
        for i in idx:
            row = {
                "label": lab,
                "step_from": int(d["step_from"][i]),
                "step_to": int(d["step_to"][i]),
                "dA": float(dA[i]),
                "Qd": float(Qd[i]),
                "violation": float(viol[i]),
            }
            for j, c in enumerate(x_cols):
                row[f"d{c}"] = float(dX[i, j])
            rows.append(row)

    if not rows:
        pd.DataFrame([], columns=["label", "step_from", "step_to", "dA", "Qd", "violation"] + [f"d{c}" for c in x_cols]).to_csv(path, index=False)
        return

    out = pd.DataFrame(rows).sort_values("violation", ascending=False).head(top_k)
    out.to_csv(path, index=False)


def run_method(
    method: str,
    df: pd.DataFrame,
    x_cols: List[str],
    arrow_col: str,
    step_col: str,
    eps: float,
    eps_arrow: float,
    require_nondegenerate: bool,
    require_indefinite: bool,
    jump_filter_frac: float,
    use_weighted: bool,
    delta_norm: str,
    delta_norm_eps: float,
    delta_norm_scope: str,
    global_scales: Optional[np.ndarray],
    whiten_mat: Optional[np.ndarray],
) -> MethodResult:
    label_data = compute_deltas(
        df, x_cols, arrow_col, step_col, eps_arrow,
        delta_norm, delta_norm_eps, delta_norm_scope, global_scales, whiten_mat
    )
    # attach keep masks
    n_steps_total = 0
    for lab, d in label_data.items():
        keep = jump_filter_mask_per_label(d["dX"], d["forward"], jump_filter_frac)
        d["keep"] = keep
        n_steps_total += len(d["dX"])
    labels = list(label_data.keys())
    n_labels = len(labels)

    masks = _all_sign_masks(len(x_cols))
    best: Optional[MethodResult] = None

    for mask in masks:
        p, q, z = _sig_counts(mask)
        if require_nondegenerate and z != 0:
            continue
        if require_indefinite and not (p > 0 and q > 0):
            continue

        pos_scale = 1.0
        if use_weighted:
            dX_all, fwd_all, keep_all = flatten_all(label_data)
            pos_scale = pick_pos_scale_minfix(dX_all, fwd_all, keep_all, mask)

        ffracs = []
        cfracs = []
        max_viols = []
        for lab in labels:
            d = label_data[lab]
            ff, cf, mv = eval_mask_for_label(d["dX"], d["forward"], d["keep"], mask, eps, pos_scale)
            ffracs.append(ff)
            cfracs.append(cf)
            max_viols.append(mv)

        # handle labels with no forward steps (shouldn't happen if arrow is good)
        cfracs_clean = [c for c in cfracs if not (isinstance(c, float) and math.isnan(c))]
        if not cfracs_clean:
            continue

        forward_frac_min = float(np.min(ffracs)) if ffracs else 0.0
        forward_frac_mean = float(np.mean(ffracs)) if ffracs else 0.0
        cone_frac_min = float(np.min(cfracs_clean))
        cone_frac_mean = float(np.mean(cfracs_clean))
        max_violation_max = float(np.nanmax(max_viols)) if max_viols else float("nan")

        cand = MethodResult(
            method=method,
            p=p, q=q, z=z,
            mask=",".join(str(int(x)) for x in mask.tolist()),
            pos_scale=float(pos_scale),
            forward_frac_min=forward_frac_min,
            forward_frac_mean=forward_frac_mean,
            cone_frac_min=cone_frac_min,
            cone_frac_mean=cone_frac_mean,
            max_Qd_violation_max=max_violation_max,
            n_labels=n_labels,
            n_steps_total=n_steps_total,
        )

        # Ranking: primary cone_frac_min, then cone_frac_mean, then prefer pos_scale closer to 1, then larger p+q
        def key(r: MethodResult):
            return (
                r.cone_frac_min,
                r.cone_frac_mean,
                -abs(r.pos_scale - 1.0),
                (r.p + r.q),
            )

        if best is None or key(cand) > key(best):
            best = cand

    if best is None:
        # produce an empty result
        return MethodResult(
            method=method,
            p=-1, q=-1, z=-1, mask="",
            pos_scale=float("nan"),
            forward_frac_min=float("nan"),
            forward_frac_mean=float("nan"),
            cone_frac_min=float("nan"),
            cone_frac_mean=float("nan"),
            max_Qd_violation_max=float("nan"),
            n_labels=n_labels,
            n_steps_total=n_steps_total,
        )

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True, help="CSV with per-step embedding")
    ap.add_argument("--arrow-col", required=True, help="Monotone arrow coordinate (used only for forward filtering)")
    ap.add_argument("--x-cols", nargs="*", default=None, help="Explicit x columns for Q (exclude arrow). If omitted, auto-detect.")
    ap.add_argument("--delta-norm", choices=["none","mad","std"], default="none",
                    help="Normalize Δx per-axis before cone tests (none|mad|std). Helps stabilize time-like axis assignment across datasets.")
    ap.add_argument("--delta-norm-scope", choices=["global","per-label"], default="global",
                    help="How to compute per-axis scales for Δx normalization: global across all forward steps or separately per label.")
    ap.add_argument("--delta-norm-eps", type=float, default=1e-12,
                    help="Floor for Δx normalization scales to avoid division by ~0.")
    ap.add_argument("--auto-null-axis", action="store_true",
                    help="Drop x-cols whose Δx std is near-zero (computed over forward steps).")
    ap.add_argument("--null-std-abs", type=float, default=1e-9,
                    help="Absolute Δx scale threshold for null-axis detection.")
    ap.add_argument("--null-std-rel", type=float, default=1e-3,
                    help="Relative Δx scale threshold vs median scale for null-axis detection.")
    ap.add_argument("--null-metric", choices=["std","mad"], default="std",
                    help="Metric used for null-axis detection over forward Δx (std or mad).")
    ap.add_argument("--delta-whiten", action="store_true",
                    help="Whiten Δx using global forward-step covariance (after optional axis drop).")
    ap.add_argument("--whiten-eps", type=float, default=1e-12,
                    help="Eigenvalue floor for whitening to avoid blow-ups.")
    ap.add_argument("--eps", type=float, default=1e-12, help="Cone epsilon: require Q(Δx) <= eps")
    ap.add_argument("--eps-arrow", type=float, default=1e-12, help="Forward epsilon: require Δarrow >= -eps_arrow")
    ap.add_argument("--require-nondegenerate", action="store_true", help="Require z=0 (diagonal masks already z=0).")
    ap.add_argument("--require-indefinite", action="store_true", help="Require p>0 and q>0.")
    ap.add_argument("--jump-filter-frac", type=float, default=0.0, help="Drop largest ||Δx|| forward steps per label (fraction).")
    ap.add_argument("--dump-violations", action="store_true", help="Write violations CSVs for best signature per method.")
    ap.add_argument("--top-k", type=int, default=50, help="Top K violations to dump.")
    args = ap.parse_args()

    df = pd.read_csv(args.embedding)
    if LABEL_COL not in df.columns:
        raise ValueError(f"expected a '{LABEL_COL}' column")

    step_col = _pick_step_col(df)
    x_cols = args.x_cols
    if not x_cols:
        x_cols = _default_x_cols(df, args.arrow_col, step_col)

    if args.arrow_col in x_cols:
        # ensure arrow not inside Q set
        x_cols = [c for c in x_cols if c != args.arrow_col]

    if len(x_cols) < 2:
        raise ValueError(f"need at least 2 x-cols, got {x_cols}")

    def compute_forward_deltas_cols(cols: List[str]) -> np.ndarray:
        all_dX = []
        for _, g in df.groupby(LABEL_COL):
            g = g.sort_values(step_col, kind="mergesort")
            X = g[cols].to_numpy(dtype=float)
            A = g[args.arrow_col].to_numpy(dtype=float)
            if len(g) < 2:
                continue
            dX = X[1:] - X[:-1]
            dA = A[1:] - A[:-1]
            forward = dA >= (-args.eps_arrow)
            if np.any(forward):
                all_dX.append(dX[forward])
        if not all_dX:
            return np.zeros((0, len(cols)))
        return np.vstack(all_dX)

    if args.auto_null_axis:
        dX_all = compute_forward_deltas_cols(x_cols)
        if dX_all.size == 0:
            raise ValueError("no forward deltas found for null-axis detection")
        if args.null_metric == "mad":
            stds = np.array([_mad(dX_all[:, j]) for j in range(dX_all.shape[1])], float)
        else:
            stds = np.nanstd(dX_all, axis=0)
        med = float(np.nanmedian(stds)) if np.isfinite(stds).any() else 0.0
        thresh = max(args.null_std_abs, args.null_std_rel * med)
        keep = [c for c, s in zip(x_cols, stds) if s > thresh]
        dropped = [c for c in x_cols if c not in keep]
        if dropped:
            print(f"[warn] dropping near-null axes (std <= {thresh:g}): {dropped}")
            x_cols = keep
        if len(x_cols) < 2:
            raise ValueError(f"need at least 2 x-cols after dropping null axes, got {x_cols}")

    print(f"[info] step col: {step_col}")
    print(f"[info] x cols used: {x_cols}")
    print(f"[info] arrow col: {args.arrow_col}")

    if args.delta_norm != "none":
        print(f"[info] Δx normalization: {args.delta_norm} (scope={args.delta_norm_scope})")

    def compute_global_scales() -> Optional[np.ndarray]:
        if args.delta_norm == "none" or args.delta_norm_scope != "global":
            return None
        dX_all = compute_forward_deltas_cols(x_cols)
        if dX_all.size == 0:
            return np.ones(len(x_cols), float)
        return compute_delta_scales(dX_all, args.delta_norm, args.delta_norm_eps)

    global_scales = compute_global_scales()

    def compute_whitener() -> Optional[np.ndarray]:
        if not args.delta_whiten:
            return None
        dX_all = compute_forward_deltas_cols(x_cols)
        if dX_all.size == 0:
            return None
        # covariance across forward deltas
        C = np.cov(dX_all, rowvar=False, bias=True)
        # symmetric eigendecomposition
        w, V = np.linalg.eigh(C)
        w = np.where(w < args.whiten_eps, args.whiten_eps, w)
        W = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        return W

    whiten_mat = compute_whitener()
    if whiten_mat is not None:
        print("[info] Δx whitening enabled")
    methods = []
    # Unfiltered, unweighted
    methods.append(("plain", 0.0, False))
    # Filtered only
    if args.jump_filter_frac > 0:
        methods.append((f"filtered_{args.jump_filter_frac:g}", args.jump_filter_frac, False))
    # Weighted only
    methods.append(("weighted", 0.0, True))
    # Filtered + weighted
    if args.jump_filter_frac > 0:
        methods.append((f"filtered_{args.jump_filter_frac:g}_weighted", args.jump_filter_frac, True))

    results: List[MethodResult] = []
    for name, jf, weighted in methods:
        res = run_method(
            method=name,
            df=df,
            x_cols=x_cols,
            arrow_col=args.arrow_col,
            step_col=step_col,
            eps=args.eps,
            eps_arrow=args.eps_arrow,
            require_nondegenerate=args.require_nondegenerate,
            require_indefinite=args.require_indefinite,
            jump_filter_frac=jf,
            use_weighted=weighted,
            delta_norm=args.delta_norm,
            delta_norm_eps=args.delta_norm_eps,
            delta_norm_scope=args.delta_norm_scope,
            global_scales=global_scales,
            whiten_mat=whiten_mat,
        )
        results.append(res)

        print(f"\n=== Best signature: method={name} (jump_filter_frac={jf}, weighted={weighted}) ===")
        print(pd.Series(res.__dict__).to_string())

        if args.dump_violations and res.p >= 0:
            mask = np.array([int(x) for x in res.mask.split(",")], dtype=int)
            out_path = f"delta_cone_violations_{name}.csv"
            dump_violations_csv(
                out_path,
                df=df,
                x_cols=x_cols,
                arrow_col=args.arrow_col,
                step_col=step_col,
                mask=mask,
                pos_scale=float(res.pos_scale),
                eps=args.eps,
                eps_arrow=args.eps_arrow,
                jump_filter_frac=jf,
                top_k=args.top_k,
                delta_norm=args.delta_norm,
                delta_norm_eps=args.delta_norm_eps,
                delta_norm_scope=args.delta_norm_scope,
                global_scales=global_scales,
                whiten_mat=whiten_mat,
            )
            print(f"[ok] wrote: {out_path}")

    out_df = pd.DataFrame([r.__dict__ for r in results])
    out_df.to_csv("delta_cone_signature_rank.csv", index=False)
    print("\n[ok] wrote: delta_cone_signature_rank.csv")
    print("[note] 'plain' reproduces the earlier 1,-1,-1 with a single violation (if present).")
    print("[note] 'weighted' adjusts pos_scale down to remove all forward-step violations with minimal change from 1.0.")
    print("[note] 'filtered_*' drops the largest ||Δx|| forward steps per label before scoring.")


if __name__ == "__main__":
    main()
