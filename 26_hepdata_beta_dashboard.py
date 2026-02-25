#!/usr/bin/env python3
"""
26_hepdata_beta_dashboard.py

Aggregate and visualize hepdata_dashi_native/*_dashi_native_metrics.csv.

Outputs:
  out/
    summary.csv
    flags.csv
    per_obs/<label>/
      beta_components.png
      beta_dist_to_fixedpoint.png
      chi2_dof_vs_iter.png
      alpha_vs_iter.png
      r_hi_vs_iter.png
      odd_even_vs_iter.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BETA_COLS = [f"b{k}" for k in range(5)]


def load_metrics_files(root: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(root.glob("*_dashi_native_metrics.csv"))
    if not files:
        raise SystemExit(f"No *_dashi_native_metrics.csv found under: {root}")
    out: Dict[str, pd.DataFrame] = {}
    for f in files:
        label = f.name.replace("_dashi_native_metrics.csv", "")
        df = pd.read_csv(f)
        if "iter" in df.columns:
            df = df.sort_values("iter").reset_index(drop=True)
        out[label] = df
    return out


def beta_matrix(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in BETA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing beta columns: {missing}")
    return df[BETA_COLS].to_numpy(dtype=float)


def dist_to_fixedpoint(B: np.ndarray) -> np.ndarray:
    b_star = B[-1]
    return np.linalg.norm(B - b_star[None, :], axis=1)


def contraction_ratios(d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # ratios r_t = d_{t+1}/d_t (ignore final step)
    return d[1:] / (d[:-1] + eps)


def near_zero_flags(df: pd.DataFrame, thresh: float) -> Dict[str, bool]:
    flags = {}
    for c in BETA_COLS:
        vals = df[c].to_numpy(dtype=float)
        flags[f"{c}_all_near0"] = bool(np.all(np.abs(vals) < thresh))
        flags[f"{c}_final_near0"] = bool(abs(vals[-1]) < thresh)
    return flags


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_series(x, ys: List[Tuple[str, np.ndarray]], title: str, xlabel: str, ylabel: str, out: Path) -> None:
    plt.figure()
    for name, y in ys:
        plt.plot(x, y, marker="o", label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(ys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="hepdata_dashi_native",
                    help="Directory containing *_dashi_native_metrics.csv")
    ap.add_argument("--out", type=str, default="hepdata_beta_dashboard_out",
                    help="Output directory")
    ap.add_argument("--near0", type=float, default=1e-8,
                    help="Threshold for 'near zero' flags on beta components")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    ensure_dir(outdir)
    per_obs_dir = outdir / "per_obs"
    ensure_dir(per_obs_dir)

    data = load_metrics_files(root)

    summary_rows = []
    flag_rows = []

    for label, df in sorted(data.items()):
        # Basic columns
        it = df["iter"].to_numpy() if "iter" in df.columns else np.arange(len(df))
        alpha = df["alpha"].to_numpy(dtype=float) if "alpha" in df.columns else None

        B = beta_matrix(df)
        d = dist_to_fixedpoint(B)
        r = contraction_ratios(d)

        # Summary stats
        row = {
            "label": label,
            "T": int(B.shape[0]),
            "iter_min": int(np.min(it)),
            "iter_max": int(np.max(it)),
            "d0": float(d[0]),
            "d_end": float(d[-1]),
            "ratio_median": float(np.median(r)) if r.size else np.nan,
            "ratio_q10": float(np.quantile(r, 0.10)) if r.size else np.nan,
            "ratio_q90": float(np.quantile(r, 0.90)) if r.size else np.nan,
        }

        # final betas
        for k, c in enumerate(BETA_COLS):
            row[f"{c}_final"] = float(B[-1, k])
            row[f"{c}_iter0"] = float(B[0, k])

        # optional metrics if present
        for col in ["chi2_dof", "odd_even_ratio", "R_E_hi", "condA"]:
            row[col + "_final"] = float(df[col].to_numpy(dtype=float)[-1]) if col in df.columns else np.nan

        summary_rows.append(row)

        flags = {"label": label, "near0_thresh": float(args.near0)}
        flags.update(near_zero_flags(df, args.near0))
        flag_rows.append(flags)

        # --- per-observable plots
        od = per_obs_dir / label
        ensure_dir(od)

        # beta components vs iter
        plot_series(
            it,
            [(c, df[c].to_numpy(dtype=float)) for c in BETA_COLS],
            title=f"{label}: beta components vs iter",
            xlabel="iter",
            ylabel="beta",
            out=od / "beta_components.png",
        )

        # dist to fixedpoint
        plot_series(
            it,
            [("||beta_t - beta_*||", d)],
            title=f"{label}: distance to fixedpoint in beta space",
            xlabel="iter",
            ylabel="L2 distance",
            out=od / "beta_dist_to_fixedpoint.png",
        )

        # chi2/dof
        if "chi2_dof" in df.columns:
            plot_series(
                it,
                [("chi2/dof", df["chi2_dof"].to_numpy(dtype=float))],
                title=f"{label}: chi2/dof vs iter",
                xlabel="iter",
                ylabel="chi2/dof",
                out=od / "chi2_dof_vs_iter.png",
            )

        # alpha schedule
        if alpha is not None:
            plot_series(
                it,
                [("alpha", alpha)],
                title=f"{label}: alpha vs iter",
                xlabel="iter",
                ylabel="alpha",
                out=od / "alpha_vs_iter.png",
            )

        # R_E_hi and odd_even_ratio if present
        if "R_E_hi" in df.columns:
            plot_series(
                it,
                [("R_E_hi", df["R_E_hi"].to_numpy(dtype=float))],
                title=f"{label}: R_E_hi vs iter",
                xlabel="iter",
                ylabel="R_E_hi",
                out=od / "r_hi_vs_iter.png",
            )

        if "odd_even_ratio" in df.columns:
            plot_series(
                it,
                [("odd_even_ratio", df["odd_even_ratio"].to_numpy(dtype=float))],
                title=f"{label}: odd_even_ratio vs iter",
                xlabel="iter",
                ylabel="odd/even",
                out=od / "odd_even_vs_iter.png",
            )

    # write outputs
    pd.DataFrame(summary_rows).sort_values("label").to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(flag_rows).sort_values("label").to_csv(outdir / "flags.csv", index=False)

    print(f"[ok] wrote: {outdir/'summary.csv'}")
    print(f"[ok] wrote: {outdir/'flags.csv'}")
    print(f"[ok] per-observable plots in: {per_obs_dir}")


if __name__ == "__main__":
    main()
