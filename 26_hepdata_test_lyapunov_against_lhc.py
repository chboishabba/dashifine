#!/usr/bin/env python3
"""
Test DASHI-style Lyapunov/energy descent on hepdata LHC runs.

What it does:
- Crawls --root for per-iteration result files.
- Extracts per-label time series: beta (b0..b4), chi2/dof, (optional) dist-to-fixedpoint d.
- Defines candidate energies E_t and tests monotone descent.
- Exports:
  * per_label_timeseries.csv
  * per_label_energy_report.csv
  * overall_certification.json
  * plots (optional)

This is intentionally defensive because different runs write different filenames.
"""

from __future__ import annotations
import argparse, json, os, re, glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


ITER_RE = re.compile(r"(?:^|[^0-9])iter(?:ation)?[_\- ]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)


def find_iter_from_path(p: str) -> Optional[int]:
    m = ITER_RE.search(os.path.basename(p)) or ITER_RE.search(p)
    if not m:
        return None
    return int(m.group(1))


def discover_candidate_tables(root: str) -> List[str]:
    """
    Look for csv/parquet/json that might contain per-iter results.
    We bias toward small-ish tables, and ones whose names mention beta/iter/chi2.
    """
    pats = [
        "**/*.csv",
        "**/*.parquet",
        "**/*.json",
    ]
    files = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))

    keep = []
    for p in files:
        name = os.path.basename(p).lower()
        if any(k in name for k in ["beta", "iter", "chi2", "dashboard", "trace", "traj", "history", "per_obs", "metrics"]):
            keep.append(p)
            continue
        # If the file path contains iterNN somewhere, also keep.
        if find_iter_from_path(p) is not None:
            keep.append(p)

    # De-dup
    keep = sorted(list(dict.fromkeys(keep)))
    return keep


def try_read_table(path: str) -> Optional[pd.DataFrame]:
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        if path.lower().endswith(".json"):
            # accept list-of-dicts or dict-of-lists
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                return pd.DataFrame(obj)
    except Exception:
        return None
    return None


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out


def extract_timeseries_from_tables(files: List[str]) -> pd.DataFrame:
    """
    Build a unified long table with columns:
      label, iter, b0..b4, chi2_dof, odd_even_ratio, R_E_hi, alpha, d(optional)
    from any tables that contain subsets of these.
    """
    rows = []

    for p in files:
        df = try_read_table(p)
        if df is None or len(df) == 0:
            continue
        df = normalize_cols(df)

        # Label can be a column, otherwise derive from filename.
        label_col = None
        for cand in ["label", "observable", "obs", "name"]:
            if cand in df.columns:
                label_col = cand
                break
        label_from_path = None
        if label_col is None:
            base = os.path.basename(p)
            label_from_path = os.path.splitext(base)[0]
            for suffix in [
                "_dashi_native_metrics",
                "_dashi_native",
                "_metrics",
                "_beta",
                "_trace",
                "_history",
            ]:
                if label_from_path.endswith(suffix):
                    label_from_path = label_from_path[: -len(suffix)]
                    break

        # Iter may be in a column or embedded in filename/path
        iter_col = "iter" if "iter" in df.columns else None
        iter_from_path = find_iter_from_path(p)

        # Identify beta columns (b0..b4) if present
        beta_cols = [c for c in df.columns if re.fullmatch(r"b[0-4]", c)]
        # Allow forms like b0_final etc; we only want per-iter
        if not beta_cols:
            beta_cols = [c for c in df.columns if re.fullmatch(r"b[0-4]\b.*", c)]

        # Identify other common fields
        def pick(*names):
            for n in names:
                if n in df.columns:
                    return n
            return None

        chi2_col = pick("chi2_dof", "chi2/dof", "chi2_dof_final", "chi2")
        odd_col  = pick("odd_even_ratio", "odd_even_ratio_final", "odd/even")
        reh_col  = pick("R_E_hi", "R_E_hi_final")
        alpha_col = pick("alpha", "alpha_final")
        d_col = pick("d", "dist", "distance", "d_to_fp", "dist_to_fixedpoint", "d_end", "d0")

        # If there is no iter column and no iter in path, skip (not a timeseries table)
        if iter_col is None and iter_from_path is None:
            continue

        for _, r in df.iterrows():
            if label_col is not None:
                label = str(r[label_col])
            else:
                label = str(label_from_path) if label_from_path else "unknown"
            it = int(r[iter_col]) if iter_col is not None and pd.notna(r[iter_col]) else iter_from_path
            if it is None:
                continue

            out = {"label": label, "iter": it, "source": os.path.relpath(p)}
            # Pull b0..b4 if present exactly
            for k in ["b0", "b1", "b2", "b3", "b4"]:
                if k in df.columns and pd.notna(r.get(k, np.nan)):
                    out[k] = float(r[k])
            # If missing but there are b0_* style cols, try best-effort
            if not all(k in out for k in ["b0","b1","b2","b3","b4"]):
                for k in ["b0","b1","b2","b3","b4"]:
                    if k not in out:
                        cands = [c for c in df.columns if c.startswith(k)]
                        # prefer exact per-iter naming if exists, else skip
                        if cands:
                            # pick smallest name
                            c = sorted(cands, key=len)[0]
                            if pd.notna(r.get(c, np.nan)):
                                out[k] = float(r[c])

            if chi2_col and pd.notna(r.get(chi2_col, np.nan)):
                out["chi2_dof"] = float(r[chi2_col])
            if odd_col and pd.notna(r.get(odd_col, np.nan)):
                out["odd_even_ratio"] = float(r[odd_col])
            if reh_col and pd.notna(r.get(reh_col, np.nan)):
                out["R_E_hi"] = float(r[reh_col])
            if alpha_col and pd.notna(r.get(alpha_col, np.nan)):
                out["alpha"] = float(r[alpha_col])
            if d_col and pd.notna(r.get(d_col, np.nan)):
                out["d"] = float(r[d_col])

            # Keep only rows that have at least some beta info
            if any(k in out for k in ["b0","b1","b2","b3","b4"]) or ("d" in out):
                rows.append(out)

    if not rows:
        raise SystemExit(
            "Could not find any per-iteration tables with label+iter. "
            "Point --root at your hepdata_dashi_native run folder (not the dashboard_out folder), "
            "or ensure your per-iter logs write CSV/Parquet with columns {label, iter, b0..b4}."
        )

    ts = pd.DataFrame(rows)

    # collapse duplicates: for same (label, iter), keep last row (arbitrary) but prefer rows with more fields
    def richness(row):
        return sum([int(pd.notna(row.get(c, np.nan))) for c in ["b0","b1","b2","b3","b4","chi2_dof","d"]])

    ts["__rich"] = ts.apply(richness, axis=1)
    ts = ts.sort_values(["label","iter","__rich","source"]).drop_duplicates(["label","iter"], keep="last")
    ts = ts.drop(columns=["__rich"]).sort_values(["label","iter"]).reset_index(drop=True)
    return ts


def energy_candidates(row: pd.Series, weights: Dict[str, float]) -> Dict[str, float]:
    """
    Several energies you can test; pick the one that corresponds to your theorem.
    """
    b = np.array([row.get("b0", np.nan), row.get("b1", np.nan), row.get("b2", np.nan), row.get("b3", np.nan), row.get("b4", np.nan)], dtype=float)
    have_beta = np.isfinite(b).all()

    E = {}

    if have_beta:
        E["E_beta_L2"] = float(np.linalg.norm(b, 2))
        E["E_beta_L1"] = float(np.linalg.norm(b, 1))
        # “projected curvature/defect proxy”: emphasize b2..b4 if that’s your interpretation
        E["E_tail"] = float(np.linalg.norm(b[2:], 1))

    if "d" in row and pd.notna(row["d"]):
        E["E_dist_to_fp"] = float(row["d"])

    # A conservative “MDL proxy” that *does not* use chi2 (since chi2 is in the wrong gauge):
    # model_bits ~ L1(beta) ; residual_bits ~ log(1 + R_E_hi) or log(1 + |odd-even|)
    # If you don’t have those, it reduces to model_bits only.
    mdl = 0.0
    if have_beta:
        mdl += weights.get("w_beta", 1.0) * float(np.linalg.norm(b, 1))
    if pd.notna(row.get("R_E_hi", np.nan)):
        mdl += weights.get("w_reh", 1.0) * float(np.log1p(abs(row["R_E_hi"])))
    if pd.notna(row.get("odd_even_ratio", np.nan)):
        mdl += weights.get("w_odd", 1.0) * float(np.log1p(abs(row["odd_even_ratio"])))
    E["E_MDL_proxy"] = float(mdl)

    # Optional “bounded tradeoff” energy: model - lambda*something
    # (use only if your theorem says projection trades residual for model length)
    if have_beta and pd.notna(row.get("chi2_dof", np.nan)):
        E["E_beta_minus_logchi2"] = float(np.linalg.norm(b, 1) - weights.get("w_chi2", 1.0) * np.log1p(row["chi2_dof"]))

    return E


def monotone_descent_fraction(x: np.ndarray, tol: float = 0.0) -> float:
    if len(x) < 2:
        return float("nan")
    ok = 0
    tot = 0
    for i in range(len(x) - 1):
        if np.isfinite(x[i]) and np.isfinite(x[i+1]):
            tot += 1
            if x[i+1] <= x[i] + tol:
                ok += 1
    return ok / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder of hepdata_dashi_native run")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--tol", type=float, default=0.0, help="Monotone tolerance")
    ap.add_argument("--w_beta", type=float, default=1.0)
    ap.add_argument("--w_reh", type=float, default=1.0)
    ap.add_argument("--w_odd", type=float, default=1.0)
    ap.add_argument("--w_chi2", type=float, default=1.0)
    ap.add_argument("--plots", action="store_true", help="Write plots (requires matplotlib)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    files = discover_candidate_tables(args.root)
    ts = extract_timeseries_from_tables(files)

    # Compute energies for every row
    weights = {"w_beta": args.w_beta, "w_reh": args.w_reh, "w_odd": args.w_odd, "w_chi2": args.w_chi2}
    energy_rows = []
    for _, r in ts.iterrows():
        E = energy_candidates(r, weights)
        out = {"label": r["label"], "iter": int(r["iter"])}
        out.update(E)
        energy_rows.append(out)

    Edf = pd.DataFrame(energy_rows).sort_values(["label","iter"]).reset_index(drop=True)

    # Merge energies back into the timeseries
    full = ts.merge(Edf, on=["label","iter"], how="left")
    full.to_csv(os.path.join(args.out, "per_label_timeseries.csv"), index=False)

    # Per-label descent report
    energy_cols = [c for c in Edf.columns if c.startswith("E_")]
    rep = []
    for label, g in Edf.groupby("label"):
        g = g.sort_values("iter")
        for ec in energy_cols:
            frac = monotone_descent_fraction(g[ec].to_numpy(dtype=float), tol=args.tol)
            rep.append({
                "label": label,
                "energy": ec,
                "iters": int(g["iter"].min()),
                "iter_max": int(g["iter"].max()),
                "monotone_fraction": float(frac),
                "start": float(g[ec].iloc[0]) if len(g) else np.nan,
                "end": float(g[ec].iloc[-1]) if len(g) else np.nan,
            })

    repdf = pd.DataFrame(rep).sort_values(["energy","label"])
    repdf.to_csv(os.path.join(args.out, "per_label_energy_report.csv"), index=False)

    # Overall certification summary
    overall = {}
    for ec in energy_cols:
        overall[ec] = float(np.nanmean(repdf[repdf["energy"] == ec]["monotone_fraction"].to_numpy(dtype=float)))

    cert = {
        "root": os.path.abspath(args.root),
        "n_rows": int(len(full)),
        "labels": sorted(full["label"].unique().tolist()),
        "energies": energy_cols,
        "mean_monotone_fraction_by_energy": overall,
        "note": (
            "If chi2/dof increases while a DASHI energy decreases, "
            "that is consistent with 'covariance-fit is not gauge-invariant under DASHI projection'. "
            "Use E_dist_to_fp / E_MDL_proxy / E_beta_L1 as primary Lyapunov candidates."
        ),
    }
    with open(os.path.join(args.out, "overall_certification.json"), "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)

    # Optional plots
    if args.plots:
        if plt is None:
            print("[warn] matplotlib not available; skipping plots")
        else:
            for ec in energy_cols:
                plt.figure()
                for label, g in Edf.groupby("label"):
                    g = g.sort_values("iter")
                    plt.plot(g["iter"], g[ec], label=label)
                plt.xlabel("iter")
                plt.ylabel(ec)
                plt.title(f"Energy trajectory: {ec}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(args.out, f"{ec}.png"), dpi=160)
                plt.close()

    print("[ok] wrote:", os.path.join(args.out, "per_label_timeseries.csv"))
    print("[ok] wrote:", os.path.join(args.out, "per_label_energy_report.csv"))
    print("[ok] wrote:", os.path.join(args.out, "overall_certification.json"))
    if args.plots and plt is not None:
        print("[ok] wrote plots:", args.out)


if __name__ == "__main__":
    main()
