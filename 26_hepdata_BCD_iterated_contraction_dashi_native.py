import os, math, csv
import numpy as np
import requests
import matplotlib.pyplot as plt
from typing import Optional, Any, Dict, List

# ===============================
# DATASETS (same 5)
# ===============================
TABLES: List[Dict[str, Any]] = [
    {"label": "z_pt_7tev_atlas", "table_url": "https://www.hepdata.net/download/table/ins1300647/Table 1/json"},
    {"label": "ttbar_mtt_8tev_cms",
     "table_url": "https://www.hepdata.net/download/table/ins1370682/Table 39/json",
     "cov_url": "https://www.hepdata.net/download/table/ins1370682/Table 40/json"},
    {"label": "hgg_pt_8tev_atlas", "table_url": "https://www.hepdata.net/download/table/ins1391147/Table 2/json"},
    {"label": "dijet_chi_7tev_cms", "table_url": "https://www.hepdata.net/download/table/ins889175/Table 1/json"},
    {"label": "dijet_chi_13tev_cms_mgt6",
     "record": "ins1663452", "table": "Table 1"},
    {"label": "atlas_4l_m4l_8tev",
     "record": "ins1394865", "table": "Table 1", "cov_table": "Table 4"},
    {"label": "atlas_4l_pt4l_8tev",
     "record": "ins1394865", "table": "Table 2", "cov_table": "Table 5"},
    {"label": "ptll_76_106_table", "table_url": "https://www.hepdata.net/record/129883?format=json"},
]

OUTDIR = "hepdata_dashi_native"
os.makedirs(OUTDIR, exist_ok=True)

# Representation: Legendre invariants up to degree 4
DEG = 4
K = DEG + 1

# Iterations of contraction flow in internal representation space
N_ITERS = 12

# Penalty schedule (projection strength): alpha_t = alpha0 * growth^t
ALPHA0 = 1e-6
ALPHA_GROWTH = 10.0

# Structured penalty weights w_k (k=0..4)
# - odd modes heavily penalized (implements involution quotient 𝒞)
# - higher order increasingly penalized (implements projection 𝒫)
ODD_MULT = 1e12
HI_ORDER_POW = 4  # growth ~ k^pow for k>=3

# Numerical stability ridge on covariance
RIDGE_COV = 1e-12

# B/C local “quadratic dominance” probe in parameter space
DIR_PROBES = 24
EPS_DIR = 1e-4

# ===============================
# HEPData download / parse
# ===============================
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
            low = parse_num(xb["low"]); high = parse_num(xb["high"])
            mid = 0.5*(low+high)
        else:
            val = parse_num(xb.get("value"))
            low = val; high = val; mid = val
        if mid <= 0:
            mid = 0.5*high
        xbins.append((low, high))
        xmid.append(mid)
        y.append(float(row["y"][0]["value"]))
    return xbins, np.array(xmid, float), np.array(y, float)

def _parse_error_val(val, yval: float) -> float:
    if isinstance(val, str) and val.strip().endswith("%"):
        pct = float(val.strip().replace("%", ""))
        return abs(yval) * pct / 100.0
    return abs(float(val))


def extract_yerr(table_json: dict):
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
    return arr if np.isfinite(arr).any() else None


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

def cov_to_logspace(Vy: np.ndarray, y: np.ndarray):
    # Cov(log y_i, log y_j) ≈ Cov(y_i,y_j)/(y_i y_j)
    return Vy / np.outer(y, y)

def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))

# ===============================
# Representation basis: Legendre invariants on x∈[-1,1]
# ===============================
def scale_to_unit(x):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax == xmin:
        return np.zeros_like(x)
    t = (x - xmin) / (xmax - xmin)
    return 2.0*t - 1.0

def design_legendre(x_unit, degree: int):
    from numpy.polynomial.legendre import legvander
    return legvander(x_unit, degree)  # columns P0..Pdeg

# ===============================
# GLS + Penalized GLS (DASHI-native contraction)
# ===============================
def regularize_cov(V):
    n = V.shape[0]
    base = np.trace(V) / max(n, 1)
    ridge = max(RIDGE_COV * base, 1e-12)
    return V + ridge*np.eye(n)

def penalized_gls_fit(X, z, V, alpha, w_diag):
    """
    Solve:
      argmin_b  (z - Xb)^T V^{-1} (z - Xb) + alpha * b^T W b
    where W = diag(w_diag) is invariant-structured penalty.

    This implements:
      - 𝒞 involution quotient: huge weights on odd coefficients
      - 𝒫 projection: increasing weights on high-order invariants
      - 𝑅 renorm: re-fit effective low-order coefficients each step
    """
    Vreg = regularize_cov(V)
    try:
        Vinv = np.linalg.inv(Vreg)
    except np.linalg.LinAlgError:
        Vinv = np.linalg.pinv(Vreg)

    XtV = X.T @ Vinv
    A = XtV @ X + alpha * np.diag(w_diag)
    rhs = XtV @ z
    beta = np.linalg.solve(A, rhs)

    resid = z - X @ beta
    chi2 = float(resid.T @ Vinv @ resid)
    return beta, resid, chi2, Vinv, Vreg, A

def aic_bic(chi2, k, n):
    aic = chi2 + 2*k
    bic = chi2 + k*math.log(n)
    return aic, bic

def hessian_effective(A):
    # For penalized GLS, the quadratic form in beta-space is A (up to constants).
    # Treat A as "effective Hessian" of the action in parameter space.
    Asym = 0.5*(A + A.T)
    evals = np.linalg.eigvalsh(Asym)
    evals = np.sort(evals)
    cond = float((evals[-1] + 1e-30) / (evals[0] + 1e-30))
    return cond, float(evals[0]), float(evals[len(evals)//2]), float(evals[-1])

# ===============================
# D: involution parity metrics
# ===============================
def odd_even_ratio(beta):
    odd = math.sqrt(beta[1]**2 + beta[3]**2) if len(beta) >= 4 else float("nan")
    even = math.sqrt(beta[0]**2 + beta[2]**2 + beta[4]**2) + 1e-12 if len(beta) >= 5 else float("nan")
    return float(odd/even)

def invariant_energy_ratio(beta):
    # R_E_hi = (E3+E4)/(E0+E1+E2)
    E = [float(b*b) for b in beta]
    num = (E[3] + E[4]) if len(E) >= 5 else float("nan")
    den = (E[0] + E[1] + E[2] + 1e-12) if len(E) >= 3 else float("nan")
    return E, float(num/den)

# ===============================
# B: local non-quadraticity probe in beta space
# ===============================
def loss_chi2(beta, X, z, Vinv):
    r = z - X @ beta
    return float(r.T @ Vinv @ r)

def taylor_coeffs_along_dir(beta, v, X, z, Vinv, eps=EPS_DIR):
    L0  = loss_chi2(beta, X, z, Vinv)
    Lp  = loss_chi2(beta + eps*v, X, z, Vinv)
    Lm  = loss_chi2(beta - eps*v, X, z, Vinv)
    Lpp = loss_chi2(beta + 2*eps*v, X, z, Vinv)
    Lmm = loss_chi2(beta - 2*eps*v, X, z, Vinv)

    c2 = (Lp + Lm - 2*L0) / (2.0 * eps**2)
    c3 = (Lpp - 2*Lp + 2*Lm - Lmm) / (2.0 * eps**3)
    c4 = (Lpp - 4*Lp + 6*L0 - 4*Lm + Lmm) / (eps**4)
    return float(c2), float(c3), float(c4)

def probe_quadraticity(beta, X, z, Vinv, n_dirs=DIR_PROBES):
    ratios_c3 = []
    ratios_c4 = []
    for _ in range(n_dirs):
        v = np.random.randn(len(beta))
        v /= (np.linalg.norm(v) + 1e-12)
        c2, c3, c4 = taylor_coeffs_along_dir(beta, v, X, z, Vinv)
        ratios_c3.append(abs(c3) / (abs(c2) + 1e-12))
        ratios_c4.append(abs(c4) / (abs(c2) + 1e-12))

    def summarise(a):
        a = np.array(a, float)
        return float(np.median(a)), float(np.quantile(a, 0.1)), float(np.quantile(a, 0.9))

    c3m, c3q10, c3q90 = summarise(ratios_c3)
    c4m, c4q10, c4q90 = summarise(ratios_c4)
    return (c3m, c3q10, c3q90), (c4m, c4q10, c4q90)

# ===============================
# DASHI-native contraction weights
# ===============================
def dashi_weights():
    """
    w_k:
      - odd k => huge penalty (involution quotient)
      - k>=3 => grows rapidly (projection onto low-order invariants)
      - k=0,2 lightly penalized (they survive)
    """
    w = np.ones(K, float)
    for k in range(K):
        if k % 2 == 1:
            w[k] *= ODD_MULT
        if k >= 3:
            w[k] *= float(k**HI_ORDER_POW)
    return w

# ===============================
# Analysis runner
# ===============================
def analyze_dataset(entry: Dict[str, Any]):
    table_id = _resolve_ref(entry, "table_url", "table")
    cov_id = _resolve_ref(entry, "cov_url", "cov_table")
    label = entry["label"]
    print(f"\n=== {label} ===")
    t = get_json(table_id)
    c = get_json(cov_id) if cov_id else None

    xbins, x, y = extract_xy(t)
    Vy = None
    if c is not None:
        Vy = extract_cov_matrix(c, xbins)
    else:
        yerr = extract_yerr(t)
        if yerr is not None:
            Vy = np.diag(yerr ** 2)
    if Vy is None:
        Vy = np.eye(len(x))

    # Work in logx/logy as you did
    lx = safe_log(x)
    z  = safe_log(y)
    Vz = cov_to_logspace(Vy, y)

    x_unit = scale_to_unit(lx)
    X = design_legendre(x_unit, DEG)

    w_diag = dashi_weights()

    csv_path = os.path.join(OUTDIR, f"{label}_dashi_native_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "iter","alpha","n",
            "chi2","chi2_dof","AIC","BIC",
            "odd_even_ratio","R_E_hi",
            "c3c2_med","c3c2_q10","c3c2_q90",
            "c4c2_med","c4c2_q10","c4c2_q90",
            "condA","eigminA","eigmedA","eigmaxA",
            "b0","b1","b2","b3","b4"
        ])

        iters = []
        alphas = []
        Rhi_series = []
        odd_series = []
        c3c2_series = []
        c4c2_series = []
        cond_series = []

        for it in range(N_ITERS+1):
            alpha = ALPHA0 * (ALPHA_GROWTH ** it)

            beta, resid, chi2, Vinv, Vreg, A = penalized_gls_fit(X, z, Vz, alpha, w_diag)
            dof = len(z) - len(beta)
            chi2dof = chi2 / max(dof, 1)

            aic, bic = aic_bic(chi2, len(beta), len(z))

            # D metrics
            odd = odd_even_ratio(beta)
            E, Rhi = invariant_energy_ratio(beta)

            # B metric
            (c3m, c3q10, c3q90), (c4m, c4q10, c4q90) = probe_quadraticity(beta, X, z, Vinv, DIR_PROBES)

            # C metric (effective action Hessian in beta space)
            condA, emin, emed, emax = hessian_effective(A)

            wr.writerow([
                it, alpha, len(z),
                chi2, chi2dof, aic, bic,
                odd, Rhi,
                c3m, c3q10, c3q90,
                c4m, c4q10, c4q90,
                condA, emin, emed, emax,
                beta[0], beta[1], beta[2], beta[3], beta[4]
            ])

            print(
                f"iter={it:2d} alpha={alpha:8.1e} chi2/dof={chi2dof:9.3f} "
                f"odd/even={odd:8.3g} R_E_hi={Rhi:8.3g} "
                f"|c3|/|c2|~{c3m:8.3g} |c4|/|c2|~{c4m:8.3g} cond(A)~{condA:8.3g}"
            )

            iters.append(it)
            alphas.append(alpha)
            Rhi_series.append(Rhi)
            odd_series.append(odd)
            c3c2_series.append(c3m)
            c4c2_series.append(c4m)
            cond_series.append(condA)

    # Plots
    def save_plot(y, title, ylabel, fname):
        plt.figure(figsize=(10,4))
        plt.plot(iters, y, marker='o')
        plt.title(f"{label}: {title}")
        plt.xlabel("iteration")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, fname), dpi=160)
        plt.close()

    save_plot(Rhi_series, "DASHI-native: high-order invariant energy ratio R_E_hi", "R_E_hi", f"{label}_Rhi.png")
    save_plot(odd_series, "DASHI-native: odd/even ratio (should go → 0)", "odd/even", f"{label}_odd_even.png")
    save_plot(c3c2_series, "B: quadratic dominance probe median |c3|/|c2|", "|c3|/|c2|", f"{label}_c3c2.png")
    save_plot(c4c2_series, "B: quadratic dominance probe median |c4|/|c2|", "|c4|/|c2|", f"{label}_c4c2.png")
    save_plot(cond_series, "C: effective action condition number cond(A)", "cond(A)", f"{label}_condA.png")

    # Also plot vs alpha on log axis (more “RG-like”)
    plt.figure(figsize=(10,4))
    plt.semilogx(alphas, Rhi_series, marker='o')
    plt.title(f"{label}: R_E_hi vs alpha (projection strength)")
    plt.xlabel("alpha")
    plt.ylabel("R_E_hi")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{label}_Rhi_vs_alpha.png"), dpi=160)
    plt.close()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots to: {OUTDIR}/")

def main():
    print(f"Output dir: {OUTDIR}")
    for entry in TABLES:
        analyze_dataset(entry)
    print("Done.")

if __name__ == "__main__":
    main()
