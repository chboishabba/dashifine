import os, math
import numpy as np
import requests
import matplotlib.pyplot as plt

# ===============================
# CONFIG: CMS DY + ≥1 jet tables
# ===============================
TABLES = [
    (129890, 129891, "pTll_50_76"),
    (129892, 129893, "pTll_76_106"),
    (129894, 129895, "pTll_106_170"),
    (129896, 129897, "pTll_170_350"),
    (129902, 129903, "phistar_50_76"),
]

BASE_URL = "https://www.hepdata.net/record/{}?format=json"

OUTDIR = "hepdata_tests_out"
os.makedirs(OUTDIR, exist_ok=True)

DEGREES = [1,2,3,4]

# ===============================
# Download / parse
# ===============================

def get_json(record_id: int) -> dict:
    url = BASE_URL.format(record_id)
    r = requests.get(url, timeout=60)
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
        raise ValueError(f"Could not find 'Total uncertainty' in headers: {headers}")
    y_col = total_idx - 2
    if y_col < 0:
        raise ValueError("Unexpected header layout for covariance table.")

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

# ===============================
# Math helpers
# ===============================

def design_matrix(x: np.ndarray, degree: int):
    return np.vstack([x**k for k in range(degree+1)]).T

def cov_to_logspace(Vy: np.ndarray, y: np.ndarray):
    # Cov(log y) ≈ J Cov(y) J with J=diag(1/y)
    return Vy / np.outer(y, y)

def gls_fit(X: np.ndarray, y: np.ndarray, V: np.ndarray, ridge_scale=1e-12):
    n = len(y)
    ridge = ridge_scale * (np.trace(V) / max(n,1))
    Vreg = V + ridge * np.eye(n)
    Vinv = np.linalg.inv(Vreg)
    XtV = X.T @ Vinv
    beta = np.linalg.solve(XtV @ X, XtV @ y)
    resid = y - X @ beta
    chi2 = float(resid.T @ Vinv @ resid)
    return beta, resid, chi2, Vinv

def aic_bic(chi2: float, k: int, n: int):
    aic = chi2 + 2*k
    bic = chi2 + k*math.log(n)
    return aic, bic

def cholesky_whitener(V: np.ndarray, ridge_scale=1e-12):
    n = V.shape[0]
    ridge = ridge_scale * (np.trace(V) / max(n,1))
    Vreg = V + ridge * np.eye(n)
    # If not SPD due to numeric issues, fall back to eigen
    try:
        L = np.linalg.cholesky(Vreg)
        # whiten: w = L^{-1} r
        Linv = np.linalg.inv(L)
        return Linv
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(Vreg)
        w = np.maximum(w, 1e-30)
        Winvhalf = np.diag(1.0/np.sqrt(w))
        return Winvhalf @ Q.T

def norm_ppf(p):
    # Acklam inverse normal CDF approximation (good enough for QQ plots)
    # Source: Peter J. Acklam approximation (public domain-ish common use)
    # Works for 0<p<1
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

# ===============================
# Projections
# ===============================

def pairwise_rebin_matrix(n: int):
    m = n//2
    A = np.zeros((m,n))
    for r in range(m):
        A[r,2*r] = 0.5
        A[r,2*r+1] = 0.5
    return A

def smooth3_matrix(n: int):
    A = np.zeros((n,n))
    A[0,0] = 1.0
    A[n-1,n-1] = 1.0
    for i in range(1,n-1):
        A[i,i-1] = 1/3
        A[i,i]   = 1/3
        A[i,i+1] = 1/3
    return A

def apply_projection(A, x, y, V):
    y2 = A @ y
    V2 = A @ V @ A.T
    x2 = A @ x
    return x2, y2, V2

def repeated_smooth(x, y, V, rounds=3):
    n = len(y)
    S = smooth3_matrix(n)
    for _ in range(rounds):
        x, y, V = apply_projection(S, x, y, V)
    return x, y, V

# ===============================
# D) Involution symmetrization on centered coordinate
# ===============================

def involution_symmetrize(x, z, V):
    """
    x is 1D coordinate (already logx if you want).
    z is dependent variable (logy).
    We center x -> xc, then create z_sym(xc_i) = 0.5*(z(xc_i) + z(-xc_i))
    using nearest-neighbor matching on -xc.

    Covariance: we approximate by symmetrizing via a linear map B (constructed).
    """
    xc = x - np.mean(x)
    n = len(xc)

    # Build nearest neighbor mapping for each i to j s.t. xc[j] ~ -xc[i]
    jmap = []
    for i in range(n):
        target = -xc[i]
        j = int(np.argmin(np.abs(xc - target)))
        jmap.append(j)

    # Linear symmetrization map: z_sym = 0.5*(I + P) z, where P swaps i -> jmap[i]
    B = np.zeros((n,n))
    for i in range(n):
        B[i,i] += 0.5
        B[i,jmap[i]] += 0.5

    zsym = B @ z
    Vsym = B @ V @ B.T
    return xc, zsym, Vsym, jmap

# ===============================
# Main per-dataset analysis + plots
# ===============================

def analyze_and_plot(table_id, cov_id, label):
    print(f"\n=== {label} ===")

    t = get_json(table_id)
    c = get_json(cov_id)

    xbins, x, y = extract_xy(t)
    Vy = extract_cov_total(c, xbins)

    # Fit coordinates
    lx = np.log(x)
    z  = np.log(y)
    Vz = cov_to_logspace(Vy, y)

    # Projections to test C:
    projections = [
        ("RAW", lx, z, Vz),
        ("REBIN2",) + apply_projection(pairwise_rebin_matrix(len(z)), lx, z, Vz),
        ("SMOOTH3x1",) + apply_projection(smooth3_matrix(len(z)), lx, z, Vz),
        ("SMOOTH3x3",) + repeated_smooth(lx, z, Vz, rounds=3),
    ]

    # Store scores and coefficients across projections
    coeffs = {deg: [] for deg in DEGREES}
    bics   = {deg: [] for deg in DEGREES}
    aics   = {deg: [] for deg in DEGREES}
    chi2d  = {deg: [] for deg in DEGREES}

    # For plotting fits for RAW only
    raw_fits = {}

    for pname, xp, zp, Vp in projections:
        n = len(zp)
        for deg in DEGREES:
            k = deg+1
            dof = n-k
            if dof <= 0:
                coeffs[deg].append([np.nan]*(k))
                bics[deg].append(np.nan)
                aics[deg].append(np.nan)
                chi2d[deg].append(np.nan)
                continue
            X = design_matrix(xp, deg)
            beta, resid, chi2, Vinv = gls_fit(X, zp, Vp)
            aic, bic = aic_bic(chi2, k, n)
            coeffs[deg].append(beta.copy())
            bics[deg].append(bic)
            aics[deg].append(aic)
            chi2d[deg].append(chi2/dof)
            if pname == "RAW":
                raw_fits[deg] = (beta, resid, chi2, Vp)

    # ============ Plot pack 1: Fits + BIC/AIC + residual diagnostics (B) ============
    fig = plt.figure(figsize=(14,10))

    # (1) Fit overlay RAW
    ax1 = fig.add_subplot(2,2,1)
    ax1.errorbar(lx, z, yerr=np.sqrt(np.diag(Vz)), fmt='o', label='data')
    x_dense = np.linspace(lx.min(), lx.max(), 400)
    for deg in DEGREES:
        beta = raw_fits[deg][0]
        yd = design_matrix(x_dense, deg) @ beta
        ax1.plot(x_dense, yd, label=f"deg={deg}")
    ax1.set_title(f"{label}: fits in (logx, logy)")
    ax1.set_xlabel("log x")
    ax1.set_ylabel("log y")
    ax1.grid(True)
    ax1.legend()

    # (2) BIC/AIC RAW bar
    ax2 = fig.add_subplot(2,2,2)
    degs = np.array(DEGREES)
    bic_raw = [bics[d][0] for d in DEGREES]
    aic_raw = [aics[d][0] for d in DEGREES]
    ax2.bar(degs-0.15, bic_raw, width=0.3, label="BIC")
    ax2.bar(degs+0.15, aic_raw, width=0.3, label="AIC")
    ax2.set_title("RAW model selection")
    ax2.set_xlabel("degree")
    ax2.set_ylabel("score (lower better)")
    ax2.grid(True, axis='y')
    ax2.legend()

    # (3) Whitened residuals vs x for best-by-BIC (B)
    ax3 = fig.add_subplot(2,2,3)
    best_deg = min(DEGREES, key=lambda d: bics[d][0])
    beta, resid, chi2, Vp = raw_fits[best_deg]
    W = cholesky_whitener(Vp)
    wres = W @ resid
    ax3.plot(lx, wres, 'o-')
    ax3.axhline(0.0)
    ax3.set_title(f"Whitened residuals vs x (best deg={best_deg})")
    ax3.set_xlabel("log x")
    ax3.set_ylabel("whitened residual")
    ax3.grid(True)

    # (4) Histogram + QQ plot
    ax4 = fig.add_subplot(2,2,4)
    ax4.hist(wres, bins=10, density=True)
    ax4.set_title("Histogram of whitened residuals")
    ax4.set_xlabel("wres")
    ax4.set_ylabel("density")
    ax4.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{label}_B_fits_residuals.png"), dpi=160)
    plt.close(fig)

    # QQ plot separate (cleaner)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(1,1,1)
    r = np.sort(wres)
    n = len(r)
    qs = np.array([norm_ppf((i+0.5)/n) for i in range(n)])
    ax.plot(qs, r, 'o')
    # reference line
    m, b = np.polyfit(qs, r, 1)
    ax.plot(qs, m*qs + b)
    ax.set_title(f"{label}: QQ plot whitened residuals (deg={best_deg})")
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Residual quantiles")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{label}_B_QQ.png"), dpi=160)
    plt.close(fig)

    # ============ Plot pack 2: C) coefficient flow under projection ============
    proj_names = [p[0] for p in projections]
    xidx = np.arange(len(proj_names))

    # plot BIC across projections
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    for deg in DEGREES:
        ax.plot(xidx, bics[deg], 'o-', label=f"deg={deg}")
    ax.set_title(f"{label}: BIC across projections (RG-ish pressure)")
    ax.set_xticks(xidx)
    ax.set_xticklabels(proj_names, rotation=20)
    ax.set_ylabel("BIC (lower better)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{label}_C_BIC_flow.png"), dpi=160)
    plt.close(fig)

    # plot coefficient magnitudes across projections (for deg=4 as example)
    # and "irrelevance" heuristic: higher-order coeffs should shrink under projection
    for deg in [2,3,4]:
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1)
        # stack coefficients per projection
        betas = coeffs[deg]
        # pad to same length
        maxk = deg+1
        for k in range(maxk):
            series = [b[k] if len(b)==maxk else np.nan for b in betas]
            ax.plot(xidx, np.abs(series), 'o-', label=f"|beta_{k}|")
        ax.set_title(f"{label}: |coefficients| flow across projections (deg={deg})")
        ax.set_xticks(xidx)
        ax.set_xticklabels(proj_names, rotation=20)
        ax.set_ylabel("|beta_k|")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"{label}_C_coeff_flow_deg{deg}.png"), dpi=160)
        plt.close(fig)

    # ============ Plot pack 3: D) involution symmetrization kills odd terms ============
    xc, zsym, Vsym, jmap = involution_symmetrize(lx, z, Vz)

    # Fit deg=4 before/after sym
    def fit_and_coeffs(xfit, zfit, Vfit, deg=4):
        X = design_matrix(xfit, deg)
        beta, resid, chi2, Vinv = gls_fit(X, zfit, Vfit)
        return beta, resid, chi2

    beta_raw4, resid_raw4, chi2_raw4 = fit_and_coeffs(lx - np.mean(lx), z, Vz, deg=4)
    beta_sym4, resid_sym4, chi2_sym4 = fit_and_coeffs(xc, zsym, Vsym, deg=4)

    # Plot odd/even coefficient magnitudes
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    idx = np.arange(5)  # beta0..beta4
    ax.bar(idx-0.15, np.abs(beta_raw4), width=0.3, label="raw |beta|")
    ax.bar(idx+0.15, np.abs(beta_sym4), width=0.3, label="sym |beta|")
    ax.set_title(f"{label}: Involution symmetrization effect on coefficients (deg=4)")
    ax.set_xlabel("coefficient index k (odd should drop under sym)")
    ax.set_ylabel("|beta_k|")
    ax.grid(True, axis='y')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{label}_D_coeff_sym.png"), dpi=160)
    plt.close(fig)

    # Plot fits before/after sym on same centered x
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    xcent = lx - np.mean(lx)
    ax.errorbar(xcent, z, yerr=np.sqrt(np.diag(Vz)), fmt='o', label='raw data')
    ax.errorbar(xc, zsym, yerr=np.sqrt(np.diag(Vsym)), fmt='x', label='sym data')
    xd = np.linspace(xcent.min(), xcent.max(), 400)
    ax.plot(xd, design_matrix(xd, 4) @ beta_raw4, label="raw fit deg=4")
    ax.plot(xd, design_matrix(xd, 4) @ beta_sym4, label="sym fit deg=4")
    ax.set_title(f"{label}: raw vs involution-symmetrized (centered logx)")
    ax.set_xlabel("centered log x")
    ax.set_ylabel("log y")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"{label}_D_fit_sym.png"), dpi=160)
    plt.close(fig)

    print(f"Saved plots for {label} -> {OUTDIR}/")

def main():
    print(f"Writing plots to: {OUTDIR}")
    for table_id, cov_id, label in TABLES:
        analyze_and_plot(table_id, cov_id, label)
    print("Done.")

if __name__ == "__main__":
    main()
