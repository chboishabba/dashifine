#!/usr/bin/env python3
"""
HEPData -> DASHI projection (10 lenses) -> ternary quantization -> activity metrics
+ optional representation-space contraction via involution-even projection.

Input: directory of .npz files, each with keys:
  - x: (N,)
  - y: (N,)
  - cov: (N,N) optional (if missing, uses diag from yerr if present)
  - yerr: (N,) optional (if cov missing)
  - name: str optional (else filename stem)

Output:
  - per-observable lens table (N x 10) in [-1,1]
  - ternary table (N x 10) in {-1,0,+1}
  - per-lens activity metrics (entropy, fliprate, var, nzfrac)
  - optional contraction iterations over representation vectors
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np

# ----------------------------
# utils
# ----------------------------

import requests

HEPDATA_TABLES: List[Dict[str, Any]] = [
    # Existing small HEPData set
    {"label": "z_pt_7tev_atlas", "table_url": "https://www.hepdata.net/download/table/ins1300647/Table 1/json"},
    {"label": "ttbar_mtt_8tev_cms",
     "table_url": "https://www.hepdata.net/download/table/ins1370682/Table 39/json",
     "cov_url": "https://www.hepdata.net/download/table/ins1370682/Table 40/json"},
    {"label": "hgg_pt_8tev_atlas", "table_url": "https://www.hepdata.net/download/table/ins1391147/Table 2/json"},
    {"label": "dijet_chi_7tev_cms", "table_url": "https://www.hepdata.net/download/table/ins889175/Table 1/json"},
    # CMS dijet angular (13 TeV) record with table selection by name
    {"label": "dijet_chi_13tev_cms_mgt6",
     "record": "ins1663452", "table": "Table 1"},
    # ATLAS 4l (8 TeV) with covariance tables
    {"label": "atlas_4l_m4l_8tev",
     "record": "ins1394865", "table": "Table 1", "cov_table": "Table 4"},
    {"label": "atlas_4l_pt4l_8tev",
     "record": "ins1394865", "table": "Table 2", "cov_table": "Table 5"},
    # pTll table JSON (76-106 GeV window)
    {"label": "ptll_76_106_table", "table_url": "https://www.hepdata.net/record/129883?format=json"},
]

def get_json(ref: str) -> dict:
    url = ref if ref.startswith("http") else f"https://www.hepdata.net/record/{ref}?format=json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _resolve_table_url(record_id: str, table_name: str) -> str:
    rec = get_json(record_id)
    for t in rec.get("data_tables", []):
        name = t.get("name") or t.get("table_name") or ""
        if name.strip() == table_name.strip():
            url = t.get("data", {}).get("json")
            if not url:
                raise ValueError(f"Table {table_name} has no JSON url in record {record_id}")
            return url
    raise ValueError(f"Table '{table_name}' not found in record {record_id}")


def _resolve_ref(entry: Dict[str, Any], key: str, table_key: str) -> Optional[str]:
    if key in entry and entry[key]:
        return entry[key]
    if entry.get("record") and entry.get(table_key):
        return _resolve_table_url(entry["record"], entry[table_key])
    return None


def extract_xy(table_json: dict):
    xbins, xmid, y = [], [], []
    def parse_num(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            s = s.replace(">", "").replace("<", "").strip()
            return float(s)
        return float(v)
    for row in table_json["values"]:
        xb = row["x"][0]
        if "low" in xb and "high" in xb:
            low = parse_num(xb["low"])
            high = parse_num(xb["high"])
            mid = 0.5 * (low + high)
        else:
            val = parse_num(xb.get("value"))
            low = val
            high = val
            mid = val
        if mid <= 0:
            mid = 0.5 * high
        xbins.append((low, high))
        xmid.append(mid)
        y.append(float(row["y"][0]["value"]))
    return xbins, np.array(xmid, float), np.array(y, float)

def _parse_error_val(val, yval: float) -> float:
    if isinstance(val, str) and val.strip().endswith("%"):
        pct = float(val.strip().replace("%", ""))
        return abs(yval) * pct / 100.0
    return abs(float(val))


def extract_yerr(table_json: dict) -> Optional[np.ndarray]:
    yerr = []
    for row in table_json["values"]:
        yval = float(row["y"][0]["value"])
        errs = row["y"][0].get("errors", [])
        if not errs:
            yerr.append(float("nan"))
            continue
        parts = []
        for e in errs:
            if "symerror" in e:
                parts.append(_parse_error_val(e["symerror"], yval))
            elif "asymerror" in e:
                ae = e["asymerror"]
                plus = _parse_error_val(ae.get("plus", 0.0), yval)
                minus = _parse_error_val(ae.get("minus", 0.0), yval)
                parts.append(max(plus, minus))
        if parts:
            yerr.append(float(np.sqrt(np.sum(np.square(parts)))))
        else:
            yerr.append(float("nan"))
    arr = np.array(yerr, float)
    if not np.isfinite(arr).any():
        return None
    return arr


def extract_cov_matrix(cov_json: dict, xbins):
    headers = [h["name"] for h in cov_json["headers"]]
    cov_idx = None
    for i, name in enumerate(headers):
        n = name.lower()
        if "covariance" in n or "cov(" in n or "matrix element" in n:
            cov_idx = i
            break
    if cov_idx is None:
        total_idx = None
        for i, name in enumerate(headers):
            if "Total uncertainty" in name:
                total_idx = i
                break
        if total_idx is None:
            raise ValueError("Covariance/Total uncertainty column not found")
        y_col = total_idx - 2
    else:
        y_col = cov_idx - 2

    n = len(xbins)
    V = np.zeros((n, n), float)
    def parse_num(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().replace(">", "").replace("<", "").strip()
            return float(s)
        return float(v)

    for row in cov_json["values"]:
        bi = (parse_num(row["x"][0]["low"]), parse_num(row["x"][0]["high"]))
        bj = (parse_num(row["x"][1]["low"]), parse_num(row["x"][1]["high"]))
        i = xbins.index(bi)
        j = xbins.index(bj)
        cov_ij = float(row["y"][y_col]["value"])
        V[i, j] = cov_ij
        V[j, i] = cov_ij
    return V


def download_hepdata_npz(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    print("Downloading HEPData and building .npz bundles...")

    for entry in HEPDATA_TABLES:
        label = entry["label"]
        table_ref = _resolve_ref(entry, "table_url", "table")
        cov_ref = _resolve_ref(entry, "cov_url", "cov_table")

        t = get_json(table_ref)
        c = get_json(cov_ref) if cov_ref else None

        xbins, x, y = extract_xy(t)
        Vy = None
        if c is not None:
            Vy = extract_cov_matrix(c, xbins)
        else:
            yerr = extract_yerr(t)
            if yerr is not None:
                Vy = np.diag(yerr ** 2)

        np.savez(
            outdir / f"{label}.npz",
            x=x,
            y=y,
            cov=Vy if Vy is not None else np.array([]),
            name=label,
        )

        print(f"  saved {label}.npz")

    print(f"All bundles written to: {outdir.resolve()}")

def safe_log(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log(np.clip(x, eps, None))

def robust_scale_to_unit(x: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """
    Robust center/scale using median/MAD, then squash to [-1,1] with tanh.
    Returns values in [-1,1].
    """
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    z = (x - med) / (1.4826 * mad)
    z = np.clip(z, -clip, clip)
    return np.tanh(z)  # already in (-1,1)

def tritize(u: np.ndarray, tau: float = 0.25) -> np.ndarray:
    """
    Map u in [-1,1] to {-1,0,+1} using threshold tau.
    """
    u = np.asarray(u, float)
    out = np.zeros_like(u, dtype=np.int8)
    out[u > tau] = 1
    out[u < -tau] = -1
    return out

def shannon_entropy_ternary(t: np.ndarray) -> float:
    """
    Entropy of ternary array with states {-1,0,+1} in nats.
    """
    t = np.asarray(t, np.int8).ravel()
    counts = np.array([(t == -1).sum(), (t == 0).sum(), (t == 1).sum()], dtype=float)
    p = counts / max(1.0, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def finite_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    First derivative dy/dx on irregular grid using centered diffs interior.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = np.full(n, np.nan, float)
    if n < 2:
        return out
    # endpoints: one-sided
    out[0] = (y[1] - y[0]) / (x[1] - x[0] + 1e-30)
    out[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2] + 1e-30)
    # interior: centered
    for i in range(1, n - 1):
        dx = x[i + 1] - x[i - 1]
        out[i] = (y[i + 1] - y[i - 1]) / (dx + 1e-30)
    return out

def second_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Second derivative d2y/dx2 on irregular grid (crude but stable enough for lensing).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = np.full(n, np.nan, float)
    if n < 3:
        return out
    # use local quadratic fit for second derivative at each interior point
    for i in range(1, n - 1):
        xs = x[i - 1:i + 2]
        ys = y[i - 1:i + 2]
        # fit a*x^2 + b*x + c
        A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
        try:
            a, b, c = np.linalg.lstsq(A, ys, rcond=None)[0]
            out[i] = 2.0 * a
        except np.linalg.LinAlgError:
            out[i] = np.nan
    out[0] = out[1]
    out[-1] = out[-2]
    return out

# ----------------------------
# polynomial fit in centered logx basis
# (used ONLY as a lens generator inside representation space)
# ----------------------------

@dataclass
class PolyFit:
    deg: int
    coeff: np.ndarray          # (deg+1,)
    yhat: np.ndarray           # (N,)
    wres: np.ndarray           # whitened residuals (N,)
    bic: float
    chi2: float
    dof: int
    condA: float

def fit_poly_centered_logx(
    x: np.ndarray,
    y: np.ndarray,
    cov: Optional[np.ndarray],
    deg: int,
) -> PolyFit:
    """
    Fit logy ~ poly(logx_centered) of degree deg using GLS with covariance (if provided),
    else WLS with diag weights.

    Returns whitened residuals via Cholesky on cov or diag.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lx = safe_log(x)
    ly = safe_log(y)

    xc = lx - np.nanmean(lx)
    N = len(x)

    # design matrix
    A = np.vstack([xc**k for k in range(deg + 1)]).T  # (N, deg+1)

    # handle covariance
    if cov is None:
        # fallback: unit weights
        W = np.eye(N)
        L = np.eye(N)
        condA = float(np.linalg.cond(A))
        beta = np.linalg.lstsq(A, ly, rcond=None)[0]
        yhat = A @ beta
        r = ly - yhat
        wres = r.copy()
        chi2 = float(r @ r)
        dof = max(1, N - (deg + 1))
    else:
        cov = np.asarray(cov, float)
        # stabilize covariance if needed
        cov = 0.5 * (cov + cov.T)
        base = np.trace(cov) / max(1, N)
        L = None
        for mult in (1e-12, 1e-9, 1e-6, 1e-3, 1e-1):
            try:
                cov_try = cov + (mult * base) * np.eye(N)
                L = np.linalg.cholesky(cov_try)
                cov = cov_try
                break
            except np.linalg.LinAlgError:
                L = None
        if L is None:
            diag = np.diag(cov)
            diag = np.where(diag > 0, diag, np.abs(diag))
            diag = diag + max(1e-6 * base, 1e-12)
            if not np.all(np.isfinite(diag)):
                diag = np.ones(N)
            cov = np.diag(diag)
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                cov = np.eye(N)
                L = np.linalg.cholesky(cov)
        # solve L z = v
        Aw = np.linalg.solve(L, A)
        yw = np.linalg.solve(L, ly)

        condA = float(np.linalg.cond(Aw))
        beta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        yhat = A @ beta
        r = ly - yhat
        wres = np.linalg.solve(L, r)  # whitened residuals
        chi2 = float(wres @ wres)
        dof = max(1, N - (deg + 1))

    # BIC for gaussian errors: chi2 + k*log(N) (up to additive constant)
    k = deg + 1
    bic = chi2 + k * math.log(max(2, N))

    return PolyFit(deg=deg, coeff=beta, yhat=yhat, wres=wres, bic=bic, chi2=chi2, dof=dof, condA=condA)

def best_poly_fit(
    x: np.ndarray,
    y: np.ndarray,
    cov: Optional[np.ndarray],
    degs=(1, 2, 3, 4),
) -> PolyFit:
    fits = [fit_poly_centered_logx(x, y, cov, d) for d in degs]
    fits.sort(key=lambda f: f.bic)
    return fits[0]

def coeff_odd_even_ratio(coeff: np.ndarray) -> float:
    """
    ratio = sum |odd coeffs| / sum |even coeffs| in centered basis.
    """
    coeff = np.asarray(coeff, float)
    odd = np.sum(np.abs(coeff[1::2]))
    even = np.sum(np.abs(coeff[0::2])) + 1e-30
    return float(odd / even)

def high_freq_energy_ratio(wres: np.ndarray) -> float:
    """
    A crude "high-frequency" ratio: energy in the top third of FFT bins / total.
    Use wres along bin index order as a proxy signal.
    """
    w = np.asarray(wres, float)
    if len(w) < 4:
        return float("nan")
    f = np.fft.rfft(w - np.mean(w))
    p = np.abs(f) ** 2
    if p.sum() <= 0:
        return 0.0
    hi_start = int(len(p) * 2 / 3)
    return float(p[hi_start:].sum() / p.sum())

# ----------------------------
# DASHI-native contraction in representation space
# ----------------------------

def even_projection_coeff(coeff: np.ndarray) -> np.ndarray:
    """
    Involution-even projection on polynomial coefficient vector:
    zero odd indices (1,3,5,...).
    """
    c = np.array(coeff, float)
    c[1::2] = 0.0
    return c

def contract_representation(coeff: np.ndarray, alpha: float) -> np.ndarray:
    """
    A simple "DASHI-native" contraction model:
    - apply even projection (involution quotient)
    - apply shrinkage that penalizes higher-order terms ~ (k^2) with strength alpha
      (this models 'only low/2nd order survives' pressure without touching observable axis)
    """
    c = even_projection_coeff(coeff)
    out = c.copy()
    for k in range(len(out)):
        out[k] = out[k] / (1.0 + alpha * (k * k))
    return out

# ----------------------------
# 10-lens map
# ----------------------------

def compute_lenses_for_observable(
    x: np.ndarray,
    y: np.ndarray,
    cov: Optional[np.ndarray],
    tau: float,
    degs=(1, 2, 3, 4),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
      U: (N,10) continuous lenses in [-1,1]
      T: (N,10) ternary lenses in {-1,0,1}
      meta: dict of useful scalar metrics
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # log domain
    lx = safe_log(x)
    ly = safe_log(y)

    # local derivatives as lenses
    slope = finite_diff(lx, ly)
    curv = second_diff(lx, ly)

    # representation fit (internal)
    pf = best_poly_fit(x, y, cov, degs=degs)
    wres = pf.wres

    # lens candidates (raw)
    # L0: level (logy)
    L0 = ly
    # L1: slope d logy / d logx
    L1 = slope
    # L2: curvature d2 logy / d logx^2
    L2 = curv
    # L3: whitened residual (pattern / mismatch)
    L3 = wres
    # L4: local roughness = |curvature|
    L4 = np.abs(curv)
    # L5: |wres| (outlierness)
    L5 = np.abs(wres)
    # L6: sign agreement between residual and curvature (structure/phase)
    L6 = np.sign(wres) * np.sign(curv)
    # L7: centered-x position (to expose symmetry breaking around center)
    xc = lx - np.mean(lx)
    L7 = xc
    # L8: local "wiggle" proxy = |Δ slope|
    L8 = np.abs(finite_diff(lx, slope))
    # L9: mirror defect proxy: compare y(x) vs y(-x) after centering in logx index order
    # crude: reverse arrays as "mirror"
    L9 = ly - ly[::-1]

    raw = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]
    U = np.column_stack([robust_scale_to_unit(v) for v in raw])
    T = np.column_stack([tritize(U[:, j], tau=tau) for j in range(U.shape[1])])

    meta = dict(
        best_deg=float(pf.deg),
        bic=float(pf.bic),
        chi2=float(pf.chi2),
        dof=float(pf.dof),
        chi2_over_dof=float(pf.chi2 / max(1, pf.dof)),
        condA=float(pf.condA),
        odd_even=float(coeff_odd_even_ratio(pf.coeff)),
        hi_energy=float(high_freq_energy_ratio(wres)),
    )
    return U, T, meta

def lens_activity(T: np.ndarray) -> Dict[str, np.ndarray]:
    """
    T: (N,10) ternary
    Returns per-lens metrics arrays of shape (10,)
    """
    T = np.asarray(T, np.int8)
    K = T.shape[1]
    ent = np.zeros(K, float)
    nzfrac = np.zeros(K, float)
    var = np.zeros(K, float)
    for k in range(K):
        tk = T[:, k]
        ent[k] = shannon_entropy_ternary(tk)
        nzfrac[k] = float((tk != 0).mean())
        var[k] = float(np.var(tk.astype(float)))
    # flip rate along bin index (how often lens changes state bin-to-bin)
    flip = np.zeros(K, float)
    for k in range(K):
        tk = T[:, k]
        if len(tk) < 2:
            flip[k] = 0.0
        else:
            flip[k] = float((tk[1:] != tk[:-1]).mean())
    return {"entropy": ent, "nzfrac": nzfrac, "var": var, "flip": flip}

# ----------------------------
# IO
# ----------------------------

def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    d = np.load(path, allow_pickle=True)
    x = d["x"]
    y = d["y"]
    cov = None
    if "cov" in d.files:
        cov = d["cov"]
        if hasattr(cov, "size") and cov.size == 0:
            cov = None
    elif "yerr" in d.files:
        yerr = d["yerr"]
        cov = np.diag(np.asarray(yerr, float) ** 2)
    name = str(d["name"]) if "name" in d.files else path.stem
    return x, y, cov, name

def save_csv(path: Path, header: List[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")

# ----------------------------
# main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                help="Download HEPData and build .npz bundles into --inp directory")
    ap.add_argument("--inp", required=True, help="Directory containing observable .npz bundles")
    ap.add_argument("--out", default="hepdata_to_dashi", help="Output directory")
    ap.add_argument("--tau", type=float, default=0.25, help="Ternary threshold on [-1,1]")
    ap.add_argument("--degs", default="1,2,3,4", help="Polynomial degrees considered for internal fit lensing")
    ap.add_argument("--contract", action="store_true", help="Also compute representation contraction over alpha sweep")
    ap.add_argument("--alpha-sweep", default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000",
                    help="Comma-separated alphas for contraction sweep (representation-space)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if args.download:
        download_hepdata_npz(inp)
    out.mkdir(parents=True, exist_ok=True)

    degs = tuple(int(s) for s in args.degs.split(",") if s.strip())
    alphas = [float(s) for s in args.alpha_sweep.split(",") if s.strip()]

    npz_files = sorted(inp.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files found in: {inp}")

    # global lens activity aggregator across observables
    global_T = []

    for f in npz_files:
        x, y, cov, name = load_npz(f)
        U, T, meta = compute_lenses_for_observable(x, y, cov, tau=args.tau, degs=degs)

        # per-observable outputs
        obs_dir = out / name
        obs_dir.mkdir(parents=True, exist_ok=True)

        # save continuous and ternary
        lens_cols = [f"L{k}" for k in range(10)]
        save_csv(obs_dir / "lenses_continuous.csv", ["bin"] + lens_cols,
                 np.column_stack([np.arange(len(x)), U]))
        save_csv(obs_dir / "lenses_ternary.csv", ["bin"] + lens_cols,
                 np.column_stack([np.arange(len(x)), T]))

        # activity
        act = lens_activity(T)
        act_rows = np.column_stack([np.arange(10), act["entropy"], act["nzfrac"], act["var"], act["flip"]])
        save_csv(obs_dir / "lens_activity.csv",
                 ["lens", "entropy_nats", "nzfrac", "var", "fliprate"],
                 act_rows)

        # meta
        meta_path = obs_dir / "meta.txt"
        with open(meta_path, "w", encoding="utf-8") as w:
            for k, v in meta.items():
                w.write(f"{k}={v}\n")

        print(f"[{name}] best_deg={int(meta['best_deg'])}  "
              f"chi2/dof={meta['chi2_over_dof']:.3f}  BIC={meta['bic']:.3f}  "
              f"odd/even={meta['odd_even']:.4g}  hiE={meta['hi_energy']:.4g}  condA={meta['condA']:.3g}")

        global_T.append(T)

        # optional: representation contraction sweep (B/D style)
        if args.contract:
            # we use the best-fit coeff vector as the "internal rep"
            pf = best_poly_fit(x, y, cov, degs=degs)
            rows = []
            for a in alphas:
                c2 = contract_representation(pf.coeff, alpha=a)
                odd_even_raw = coeff_odd_even_ratio(pf.coeff)
                odd_even_proj = coeff_odd_even_ratio(even_projection_coeff(pf.coeff))
                odd_even_c = coeff_odd_even_ratio(c2)

                rows.append([
                    a,
                    odd_even_raw,
                    odd_even_proj,
                    odd_even_c,
                    float(np.linalg.norm(pf.coeff)),
                    float(np.linalg.norm(c2)),
                ])
            rows = np.array(rows, float)
            save_csv(obs_dir / "contract_sweep.csv",
                     ["alpha", "odd_even_raw", "odd_even_evenproj", "odd_even_contracted",
                      "norm_raw", "norm_contracted"],
                     rows)

    # global activity summary across all observables/bins
    GT = np.vstack(global_T)  # (sumN,10)
    gact = lens_activity(GT)
    g_rows = np.column_stack([np.arange(10), gact["entropy"], gact["nzfrac"], gact["var"], gact["flip"]])
    save_csv(out / "GLOBAL_lens_activity.csv",
             ["lens", "entropy_nats", "nzfrac", "var", "fliprate"],
             g_rows)

    print(f"\nWrote outputs to: {out.resolve()}")
    print("Global lens activity saved to GLOBAL_lens_activity.csv")


if __name__ == "__main__":
    main()
