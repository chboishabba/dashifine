import os, math, csv
import numpy as np
import requests
import matplotlib.pyplot as plt

# ===============================
# CONFIG: HEPData tables
# ===============================
TABLES = [
    (129890, 129891, "pTll_50_76"),
    (129892, 129893, "pTll_76_106"),
    (129894, 129895, "pTll_106_170"),
    (129896, 129897, "pTll_170_350"),
    (129902, 129903, "phistar_50_76"),
]
BASE_URL = "https://www.hepdata.net/record/{}?format=json"

OUTDIR = "hepdata_BCD"
os.makedirs(OUTDIR, exist_ok=True)

# Model family for internal representation
MAX_DEG = 4
DEGREES = [1, 2, 3, 4]  # for winner tracking

# Iterated contraction schedule
N_ITERS = 10
SMOOTH_EACH = 1
REBIN_EVERY = 3

# Numerical stability
RIDGE_SCALE = 1e-12

# Loss-direction probe
DIR_PROBES = 24          # number of random directions to probe each iter
EPS_DIR = 1e-4           # step size for finite differences

# ===============================
# Download / parse helpers
# ===============================

def get_json(record_id: int) -> dict:
    r = requests.get(BASE_URL.format(record_id), timeout=60)
    r.raise_for_status()
    return r.json()

def extract_xy(table_json: dict):
    xbins, xmid, y = [], [], []
    for row in table_json["values"]:
        xb = row["x"][0]
        low = float(xb["low"])
        high = float(xb["high"])
        mid = 0.5*(low+high)
        if mid <= 0:
            mid = 0.5*high
        xbins.append((low,high))
        xmid.append(mid)
        y.append(float(row["y"][0]["value"]))
    return xbins, np.array(xmid, float), np.array(y, float)

def extract_cov_total(cov_json: dict, xbins):
    headers = [h["name"] for h in cov_json["headers"]]
    total_idx = None
    for i, name in enumerate(headers):
        if "Total uncertainty" in name:
            total_idx = i
            break
    if total_idx is None:
        raise ValueError("Total uncertainty column not found")
    y_col = total_idx - 2
    n = len(xbins)
    V = np.zeros((n,n), float)
    for row in cov_json["values"]:
        bi = (float(row["x"][0]["low"]), float(row["x"][0]["high"]))
        bj = (float(row["x"][1]["low"]), float(row["x"][1]["high"]))
        i = xbins.index(bi)
        j = xbins.index(bj)
        cov_ij = float(row["y"][y_col]["value"])
        V[i,j] = cov_ij
        V[j,i] = cov_ij
    return V

def cov_to_logspace(Vy: np.ndarray, y: np.ndarray):
    # Var(log y_i) ≈ Var(y_i) / y_i^2, Cov(log y_i, log y_j) ≈ Cov(y_i,y_j)/(y_i y_j)
    return Vy / np.outer(y, y)

# ===============================
# Bases: polynomial vs invariants
# We use Legendre basis on scaled x in [-1,1] to avoid coordinate bias.
# ===============================

def scale_to_unit_interval(x):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax == xmin:
        return np.zeros_like(x)
    t = (x - xmin) / (xmax - xmin)
    return 2.0*t - 1.0  # [-1,1]

def design_legendre(x_unit, degree: int):
    """
    Legendre Vandermonde: columns P_0..P_degree at x_unit in [-1,1]
    Implemented via numpy.polynomial.legendre.legvander
    """
    from numpy.polynomial.legendre import legvander
    return legvander(x_unit, degree)

# ===============================
# GLS fit + derived objects
# ===============================

def gls_fit(X, z, V):
    n = len(z)
    ridge = RIDGE_SCALE * (np.trace(V) / max(n,1))
    Vreg = V + ridge*np.eye(n)
    Vinv = np.linalg.inv(Vreg)

    XtV = X.T @ Vinv
    beta = np.linalg.solve(XtV @ X, XtV @ z)

    resid = z - X @ beta
    chi2 = float(resid.T @ Vinv @ resid)
    return beta, resid, chi2, Vinv, Vreg

def aic_bic(chi2: float, k: int, n: int):
    aic = chi2 + 2*k
    bic = chi2 + k*math.log(n)
    return aic, bic

def hessian_gls(X, Vinv):
    # Hessian of chi2 wrt beta is 2 X^T Vinv X
    return 2.0 * (X.T @ Vinv @ X)

def safe_eigvals_sym(A):
    # symmetric eigvals; A should be symmetric
    w = np.linalg.eigvalsh(0.5*(A + A.T))
    return np.sort(w)

# ===============================
# Contraction operators on (x, z, V)
# ===============================

def smooth3_matrix(n: int):
    A = np.zeros((n,n))
    A[0,0] = 1.0
    A[n-1,n-1] = 1.0
    for i in range(1, n-1):
        A[i,i-1] = 1/3
        A[i,i]   = 1/3
        A[i,i+1] = 1/3
    return A

def pairwise_rebin_matrix(n: int):
    m = n//2
    A = np.zeros((m,n))
    for r in range(m):
        A[r,2*r] = 0.5
        A[r,2*r+1] = 0.5
    return A

def apply_linear(A, x, z, V):
    x2 = A @ x
    z2 = A @ z
    V2 = A @ V @ A.T
    return x2, z2, V2

def contraction_step(x, z, V, step_index):
    # smooth
    for _ in range(SMOOTH_EACH):
        S = smooth3_matrix(len(z))
        x, z, V = apply_linear(S, x, z, V)

    # occasional rebin
    if REBIN_EVERY > 0 and (step_index+1) % REBIN_EVERY == 0:
        n = len(z)
        if n >= 6 and (n//2) >= 4:
            R = pairwise_rebin_matrix(n)
            x, z, V = apply_linear(R, x, z, V)

    return x, z, V

# ===============================
# D: explicit involution quotient (even projection on centered x)
# ===============================

def even_projection_matrix(x):
    """
    Build B_even that maps z -> z_even where z_even(i) = 0.5(z(i)+z(j))
    with x_j nearest to -x_i. Uses centered x.
    """
    xc = x - np.mean(x)
    n = len(xc)
    jmap = []
    for i in range(n):
        target = -xc[i]
        j = int(np.argmin(np.abs(xc - target)))
        jmap.append(j)
    B = np.zeros((n,n))
    for i in range(n):
        B[i,i] += 0.5
        B[i,jmap[i]] += 0.5
    return xc, B

def odd_even_ratio(beta):
    # for deg=4 Legendre/P basis: treat indices 1 and 3 as "odd", 0,2,4 as "even"
    # This is exactly true for centered parity coordinate and Legendre basis.
    odd = math.sqrt(beta[1]**2 + beta[3]**2)
    even = math.sqrt(beta[0]**2 + beta[2]**2 + beta[4]**2) + 1e-12
    return float(odd/even)

# ===============================
# B/C: loss-direction probe (quadratic vs cubic/quartic)
# ===============================

def loss_chi2(beta, X, z, Vinv):
    r = z - X @ beta
    return float(r.T @ Vinv @ r)

def taylor_coeffs_along_direction(beta, v, X, z, Vinv, eps=EPS_DIR):
    """
    Estimate coefficients of L(eps) = a + b e + c2 e^2 + c3 e^3 + c4 e^4 + ...
    around eps=0 using symmetric finite differences.

    We compute L at e = 0, ±eps, ±2eps.

    c2 approx: (L+ + L- - 2L0) / (2 eps^2)  [note: factor depends on definition; we keep consistent ratios]
    c3 approx: (L(2)-2L(1)+2L(-1)-L(-2)) / (2 eps^3)
    c4 approx: (L(2) - 4L(1) + 6L(0) - 4L(-1) + L(-2)) / (eps^4)
    """
    L0  = loss_chi2(beta, X, z, Vinv)
    Lp  = loss_chi2(beta + eps*v, X, z, Vinv)
    Lm  = loss_chi2(beta - eps*v, X, z, Vinv)
    Lpp = loss_chi2(beta + 2*eps*v, X, z, Vinv)
    Lmm = loss_chi2(beta - 2*eps*v, X, z, Vinv)

    c2 = (Lp + Lm - 2*L0) / (2.0 * eps**2)
    c3 = (Lpp - 2*Lp + 2*Lm - Lmm) / (2.0 * eps**3)
    c4 = (Lpp - 4*Lp + 6*L0 - 4*Lm + Lmm) / (eps**4)

    return float(c2), float(c3), float(c4)

def probe_quadratic_dominance(beta, X, z, Vinv, n_dirs=DIR_PROBES):
    ratios_c3 = []
    ratios_c4 = []
    c2s = []
    for _ in range(n_dirs):
        v = np.random.randn(len(beta))
        v /= (np.linalg.norm(v) + 1e-12)
        c2, c3, c4 = taylor_coeffs_along_direction(beta, v, X, z, Vinv)
        c2s.append(abs(c2))
        ratios_c3.append(abs(c3) / (abs(c2) + 1e-12))
        ratios_c4.append(abs(c4) / (abs(c2) + 1e-12))

    # summarize robustly
    def summarize(arr):
        arr = np.array(arr, float)
        return float(np.median(arr)), float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.9))

    med_c3, q10_c3, q90_c3 = summarize(ratios_c3)
    med_c4, q10_c4, q90_c4 = summarize(ratios_c4)
    med_c2, q10_c2, q90_c2 = summarize(c2s)

    return {
        "c2_med": med_c2, "c2_q10": q10_c2, "c2_q90": q90_c2,
        "c3_over_c2_med": med_c3, "c3_over_c2_q10": q10_c3, "c3_over_c2_q90": q90_c3,
        "c4_over_c2_med": med_c4, "c4_over_c2_q10": q10_c4, "c4_over_c2_q90": q90_c4,
    }

# ===============================
# Main analysis
# ===============================

def analyze_dataset(table_id, cov_id, label):
    print(f"\n=== {label} ===")

    t = get_json(table_id)
    c = get_json(cov_id)

    xbins, x, y = extract_xy(t)
    Vy = extract_cov_total(c, xbins)

    # Log coordinates (your working space)
    lx = np.log(x)
    z  = np.log(y)
    Vz = cov_to_logspace(Vy, y)

    # Output CSV with B/C/D metrics
    csv_path = os.path.join(OUTDIR, f"{label}_BCD_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iter","n",
            "best_deg_BIC","best_BIC","best_chi2dof",

            # D: invariant energy in Legendre basis (deg=4)
            "E0","E1","E2","E3","E4","R_E_hi",

            # D: odd/even ratio raw vs even-projected (deg=4)
            "odd_even_raw","odd_even_evenproj",

            # B/C: loss-direction probes (quadratic dominance)
            "c3_over_c2_med","c3_over_c2_q10","c3_over_c2_q90",
            "c4_over_c2_med","c4_over_c2_q10","c4_over_c2_q90",

            # C: Hessian spectrum flow (deg=4)
            "H_cond","H_eig_min","H_eig_med","H_eig_max"
        ])

        iters = []
        best_deg_series = []
        Rhi_series = []
        odd_raw_series = []
        odd_even_series = []
        c3c2_series = []
        c4c2_series = []
        Hcond_series = []

        for it in range(N_ITERS+1):
            n = len(z)

            # Winner tracking by BIC across degrees in the invariant (Legendre) basis
            x_unit = scale_to_unit_interval(lx)
            fits = {}
            for deg in DEGREES:
                k = deg + 1
                dof = n - k
                if dof <= 0:
                    continue
                X = design_legendre(x_unit, deg)
                beta, resid, chi2, Vinv, Vreg = gls_fit(X, z, Vz)
                _, bic = aic_bic(chi2, k, n)
                fits[deg] = (beta, chi2, bic, dof)

            if not fits:
                break

            best_deg = min(fits.keys(), key=lambda d: fits[d][2])
            beta_b, chi2_b, bic_b, dof_b = fits[best_deg]
            chi2dof = chi2_b / dof_b

            # Now compute B/C/D probes on deg=4, if possible
            if n - (MAX_DEG+1) <= 0:
                # too few points to fit deg=4
                E = [float("nan")]*5
                R_E = float("nan")
                odd_raw = float("nan")
                odd_even = float("nan")
                cprobe = {
                    "c3_over_c2_med": float("nan"), "c3_over_c2_q10": float("nan"), "c3_over_c2_q90": float("nan"),
                    "c4_over_c2_med": float("nan"), "c4_over_c2_q10": float("nan"), "c4_over_c2_q90": float("nan"),
                }
                H_cond = float("nan")
                Hmin = Hmed = Hmax = float("nan")
            else:
                X4 = design_legendre(x_unit, MAX_DEG)
                beta4, resid4, chi2_4, Vinv4, Vreg4 = gls_fit(X4, z, Vz)

                # D: invariant energy by order (Legendre coefficients)
                E = [float(beta4[k]**2) for k in range(5)]
                R_E = float((E[3] + E[4]) / (E[0] + E[1] + E[2] + 1e-12))

                # D: odd/even ratio raw vs explicit involution projection
                # raw: compute in centered parity coordinate (use centered lx -> scale -> Legendre)
                odd_raw = odd_even_ratio(beta4)

                xc, B = even_projection_matrix(lx)
                zE = B @ z
                VE = B @ Vz @ B.T
                xc_unit = scale_to_unit_interval(xc)
                X4E = design_legendre(xc_unit, MAX_DEG)
                beta4E, resid4E, chi2_4E, Vinv4E, Vreg4E = gls_fit(X4E, zE, VE)
                odd_even = odd_even_ratio(beta4E)

                # B/C: loss-direction quadratic dominance probes
                cprobe = probe_quadratic_dominance(beta4, X4, z, Vinv4, n_dirs=DIR_PROBES)

                # C: Hessian spectrum flow
                H = hessian_gls(X4, Vinv4)
                evals = safe_eigvals_sym(H)
                # condition number with small floor
                H_cond = float((evals[-1] + 1e-30) / (evals[0] + 1e-30))
                Hmin = float(evals[0])
                Hmed = float(evals[len(evals)//2])
                Hmax = float(evals[-1])

            # Write row
            w.writerow([
                it, n,
                best_deg, bic_b, chi2dof,
                *E, R_E,
                odd_raw, odd_even,
                cprobe.get("c3_over_c2_med", float("nan")),
                cprobe.get("c3_over_c2_q10", float("nan")),
                cprobe.get("c3_over_c2_q90", float("nan")),
                cprobe.get("c4_over_c2_med", float("nan")),
                cprobe.get("c4_over_c2_q10", float("nan")),
                cprobe.get("c4_over_c2_q90", float("nan")),
                H_cond, Hmin, Hmed, Hmax
            ])

            # Print concise line
            print(
                f"iter={it:2d} n={n:2d} best_deg={best_deg} chi2/dof={chi2dof:8.3f} BIC={bic_b:10.3f} "
                f"R_E_hi={R_E:8.3g} odd/even raw→evenproj={odd_raw:8.3g}→{odd_even:8.3g} "
                f"|c3|/|c2|~{cprobe.get('c3_over_c2_med', float('nan')):8.3g} "
                f"|c4|/|c2|~{cprobe.get('c4_over_c2_med', float('nan')):8.3g} "
                f"cond(H)~{H_cond:8.3g}"
            )

            # Store series for plots
            iters.append(it)
            best_deg_series.append(best_deg)
            Rhi_series.append(R_E)
            odd_raw_series.append(odd_raw)
            odd_even_series.append(odd_even)
            c3c2_series.append(cprobe.get("c3_over_c2_med", float("nan")))
            c4c2_series.append(cprobe.get("c4_over_c2_med", float("nan")))
            Hcond_series.append(H_cond)

            # Apply contraction for next iteration
            if it < N_ITERS:
                lx, z, Vz = contraction_step(lx, z, Vz, it)

    # Plots
    def save_plot(series, title, ylabel, fname):
        plt.figure(figsize=(10,4))
        plt.plot(iters, series, marker='o')
        plt.title(f"{label}: {title}")
        plt.xlabel("iteration")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, fname), dpi=160)
        plt.close()

    save_plot(best_deg_series, "best degree (BIC) in Legendre basis", "best_deg", f"{label}_bestdeg.png")
    save_plot(Rhi_series, "R_E_hi = (E3+E4)/(E0+E1+E2)", "R_E_hi", f"{label}_R_E_hi.png")

    plt.figure(figsize=(10,4))
    plt.plot(iters, odd_raw_series, marker='o', label="odd/even (raw deg=4)")
    plt.plot(iters, odd_even_series, marker='o', label="odd/even (even-proj then fit)")
    plt.title(f"{label}: parity quotient test (deg=4)")
    plt.xlabel("iteration")
    plt.ylabel("odd/even ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{label}_odd_even.png"), dpi=160)
    plt.close()

    save_plot(c3c2_series, "quadratic dominance probe: median |c3|/|c2|", "|c3|/|c2|", f"{label}_c3c2.png")
    save_plot(c4c2_series, "quadratic dominance probe: median |c4|/|c2|", "|c4|/|c2|", f"{label}_c4c2.png")
    save_plot(Hcond_series, "Hessian condition number cond(H) (deg=4)", "cond(H)", f"{label}_Hcond.png")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots to: {OUTDIR}/")

def main():
    print(f"Output dir: {OUTDIR}")
    for table_id, cov_id, label in TABLES:
        analyze_dataset(table_id, cov_id, label)
    print("Done.")

if __name__ == "__main__":
    main()
