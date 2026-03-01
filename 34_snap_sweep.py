#!/usr/bin/env python3
"""
34_snap_sweep.py

Quota-preserving snap sweep: keep a fraction of steps per label+direction
based on a snap score, but never drop below min_steps per direction.

Outputs:
  - snap_sweep_summary.csv
  - snap_sweep_per_label.csv
  - snap_sweep_coverage.csv
  - snap_sweep_loss_curve.csv
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

LABEL_COL = "label"
ID_COL_CANDIDATES = ["step", "iter", "t", "time", "k"]


def _pick_step_col(df: pd.DataFrame) -> str:
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"no step/iter column found among {ID_COL_CANDIDATES}")


def _parse_list(val: str, cast=float):
    if val is None or val == "":
        return []
    return [cast(v.strip()) for v in val.split(",") if v.strip() != ""]


def _rank_transform(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--x-cols", nargs="+", required=True)
    ap.add_argument("--mask", required=True, help="comma-separated +/-1")
    ap.add_argument("--mdl-quantile", type=float, default=0.40)
    ap.add_argument("--min-steps-per-label", type=int, default=3)
    ap.add_argument("--keep-fracs", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3")
    ap.add_argument("--pos-scale-min", type=float, default=0.1)
    ap.add_argument("--pos-scale-max", type=float, default=0.3)
    ap.add_argument("--pos-scale-step", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--score", choices=["beta_norm", "beta_norm_chi2"], default="beta_norm")
    ap.add_argument("--chi2-col", default="chi2_dof")
    ap.add_argument("--mdl-fallback-mode", choices=["neglog", "neglog1p"], default="neglog1p")
    ap.add_argument("--out-prefix", default="snap_sweep")
    args = ap.parse_args()

    keep_fracs = _parse_list(args.keep_fracs, float)
    if not keep_fracs:
        raise ValueError("keep-fracs is empty")

    mask = np.array([int(v) for v in args.mask.split(",")], dtype=int)
    if not np.all(np.isin(mask, [-1, 1])):
        raise ValueError("mask must be +/-1 only")

    embed = pd.read_csv(args.embedding)
    ts = pd.read_csv(args.timeseries)
    step_col = _pick_step_col(embed)

    # prepare per-label data
    label_data = {}
    for lab, g in embed.groupby(LABEL_COL):
        g = g.sort_values(step_col)
        if len(g) < 2:
            continue
        mg = ts[ts[LABEL_COL] == lab].sort_values(step_col)
        if len(mg) != len(g):
            raise ValueError(f"length mismatch for label {lab}")

        X = g[args.x_cols].to_numpy(float)
        dX = X[1:] - X[:-1]

        chi = mg[args.chi2_col].to_numpy(float)
        chi = np.maximum(chi, 1e-12)
        if args.mdl_fallback_mode == "neglog":
            mdl = -np.log(chi)
        else:
            mdl = -np.log1p(chi)

        dM = mdl[1:] - mdl[:-1]
        q = args.mdl_quantile
        lo = np.nanquantile(dM, q)
        hi = np.nanquantile(dM, 1.0 - q)
        forward = dM <= lo
        backward = dM >= hi

        # snap score (per step)
        B = mg[["b0", "b1", "b2", "b3", "b4"]].to_numpy(float)
        dB = B[1:] - B[:-1]
        dB_norm = np.linalg.norm(dB, axis=1)
        if args.score == "beta_norm_chi2":
            chi_ratio = chi[1:] / np.maximum(chi[:-1], 1e-12)
            score = dB_norm * chi_ratio
        else:
            score = dB_norm

        label_data[lab] = {
            "dX": dX,
            "forward": forward,
            "backward": backward,
            "score": score,
        }

    pos_scales = np.round(
        np.arange(args.pos_scale_min, args.pos_scale_max + 1e-12, args.pos_scale_step), 12
    )

    summary_rows = []
    per_label_rows = []
    coverage_rows = []
    loss_rows = []

    for keep_frac in keep_fracs:
        per_label_fracs = []
        per_label_cov = []

        # precompute kept indices per label/dir
        kept = {}
        for lab, d in label_data.items():
            fwd = np.where(d["forward"])[0]
            bwd = np.where(d["backward"])[0]
            if len(fwd) == 0 or len(bwd) == 0:
                kept[lab] = None
                continue

            def pick(idxs):
                if len(idxs) == 0:
                    return np.array([], dtype=int)
                # keep lowest scores
                scores = d["score"][idxs]
                k = int(np.floor(len(idxs) * keep_frac))
                k = max(k, args.min_steps_per_label)
                k = min(k, len(idxs))
                order = np.argsort(scores)
                return idxs[order[:k]]

            kf = pick(fwd)
            kb = pick(bwd)
            kept[lab] = (kf, kb)
            coverage_rows.append({
                "keep_frac": keep_frac,
                "label": lab,
                "n_forward": len(fwd),
                "n_backward": len(bwd),
                "n_forward_kept": len(kf),
                "n_backward_kept": len(kb),
            })

        for pos_scale in pos_scales:
            for lab, d in label_data.items():
                pair = kept.get(lab)
                if pair is None:
                    per_label_fracs.append(np.nan)
                    continue
                idx_f, idx_b = pair
                if len(idx_f) < args.min_steps_per_label or len(idx_b) < args.min_steps_per_label:
                    per_label_fracs.append(np.nan)
                    continue

                dx = d["dX"]
                def qvals(idxs):
                    v = dx[idxs]
                    sq = v * v
                    sp = np.sum(sq[:, mask == 1], axis=1) if np.any(mask == 1) else np.zeros(len(v))
                    sn = np.sum(sq[:, mask == -1], axis=1) if np.any(mask == -1) else np.zeros(len(v))
                    return pos_scale * sp - sn

                qf = qvals(idx_f)
                qb = qvals(idx_b)
                ok = np.concatenate([qf <= 1e-12, qb <= 1e-12])
                frac = float(np.mean(ok))
                per_label_rows.append({
                    "keep_frac": keep_frac,
                    "pos_scale": pos_scale,
                    "label": lab,
                    "cone_frac": frac,
                })
                per_label_fracs.append(frac)

            # per keep_frac + pos_scale aggregate
            if per_label_fracs:
                overall_min = float(np.nanmin(per_label_fracs))
                overall_mean = float(np.nanmean(per_label_fracs))
            else:
                overall_min = float("nan")
                overall_mean = float("nan")
            summary_rows.append({
                "keep_frac": keep_frac,
                "pos_scale": pos_scale,
                "cone_frac_min": overall_min,
                "cone_frac_mean": overall_mean,
            })

        # interval per keep_frac
        dfk = pd.DataFrame([r for r in summary_rows if r["keep_frac"] == keep_frac])
        ok = dfk["cone_frac_min"] >= args.threshold
        if ok.any():
            # compute longest contiguous interval in pos_scale order
            ps = dfk["pos_scale"].to_numpy()
            best_start = best_end = ps[ok.argmax()]
            max_len = 0.0
            start = None
            for i, good in enumerate(ok):
                if good and start is None:
                    start = ps[i]
                if (not good or i == len(ok) - 1) and start is not None:
                    end = ps[i] if good and i == len(ok) - 1 else ps[i - 1]
                    if (end - start) > max_len:
                        max_len = end - start
                        best_start, best_end = start, end
                    start = None
        else:
            best_start = best_end = np.nan
            max_len = 0.0

        # coverage summary
        cov = pd.DataFrame([r for r in coverage_rows if r["keep_frac"] == keep_frac])
        n_labels = int((cov["n_forward_kept"] >= args.min_steps_per_label).sum()) if not cov.empty else 0
        loss_rows.append({
            "keep_frac": keep_frac,
            "n_labels": n_labels,
            "mean_fwd_kept": float(cov["n_forward_kept"].mean()) if not cov.empty else float("nan"),
            "mean_bwd_kept": float(cov["n_backward_kept"].mean()) if not cov.empty else float("nan"),
            "interval_start": best_start,
            "interval_end": best_end,
            "interval_len": max_len,
        })

    # write outputs
    pd.DataFrame(summary_rows).to_csv(f"{args.out_prefix}_summary.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(f"{args.out_prefix}_per_label.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(f"{args.out_prefix}_coverage.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(f"{args.out_prefix}_loss_curve.csv", index=False)

    print(f"Wrote {args.out_prefix}_summary.csv, {args.out_prefix}_per_label.csv, {args.out_prefix}_coverage.csv, {args.out_prefix}_loss_curve.csv")


if __name__ == "__main__":
    main()
