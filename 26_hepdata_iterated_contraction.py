import os, math, csv
import numpy as np
import requests
import matplotlib.pyplot as plt

# ===============================
# CONFIG: 5 HEPData tables
# ===============================
TABLES = [
    (129890, 129891, "pTll_50_76"),
    (129892, 129893, "pTll_76_106"),
    (129894, 129895, "pTll_106_170"),
    (129896, 129897, "pTll_170_350"),
    (129902, 129903, "phistar_50_76"),
]

BASE_URL = "https://www.hepdata.net/record/{}?format=json"

OUTDIR = "hepdata_iter_contract"
os.makedirs(OUTDIR, exist_ok=True)

DEGREES = [1, 2, 3, 4]

# contraction schedule
N_ITERS = 10          # number of contraction steps
SMOOTH_EACH = 1       # apply 3-pt smooth this many times per step
REBIN_EVERY = 3       # every k steps, apply pairwise rebin (if possible)

RIDGE_SCALE = 1e-12

# ===============================
# Download / parse
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

# ===============================
# Linear algebra / fits
# ===============================

def design_matrix(x: np.ndarray, degree: int):
    return np.vstack([x**k for k in range(degree+1)]).T

def cov_to_logspace(Vy: np.ndarray, y: np.ndarray):
    return Vy / np.outer(y, y)

def gls_fit(x, z, V, deg):
    n = len(z)
    ridge = RIDGE_SCALE * (np.trace(V) / max(n,1))
    Vreg = V + ridge*np.eye(n)
    Vinv = np.linalg.inv(Vreg)
    X = design_matrix(x, deg)
    XtV = X.T @ Vinv
    beta = np.linalg.solve(XtV @ X, XtV @ z)
    resid = z - X @ beta
    chi2 = float(resid.T @ Vinv @ resid)
    return beta, resid, chi2, Vinv, Vreg

def aic_bic(chi2: float, k: int, n: int):
    aic = chi2 + 2*k
    bic = chi2 + k*math.log(n)
    return aic, bic

def cholesky_whitener(V):
    n = V.shape[0]
    ridge = RIDGE_SCALE * (np.trace(V) / max(n,1))
    Vreg = V + ridge*np.eye(n)
    try:
        L = np.linalg.cholesky(Vreg)
        return np.linalg.inv(L)
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(Vreg)
        w = np.maximum(w, 1e-30)
        return np.diag(1.0/np.sqrt(w)) @ Q.T

# ===============================
# Simple normality + moments (no scipy)
# ===============================

def moments(x):
    x = np.asarray(x)
    m = float(np.mean(x))
    v = float(np.mean((x-m)**2))
    s = float(np.mean((x-m)**3)) / (v**1.5 + 1e-30)
    k = float(np.mean((x-m)**4)) / (v**2 + 1e-30) - 3.0
    return m, v, s, k

def jarque_bera(x):
    # JB = n/6 * (S^2 + (K^2)/4)
    x = np.asarray(x)
    n = len(x)
    _, _, S, K = moments(x)
    JB = (n/6.0) * (S*S + 0.25*K*K)
    # p-value approx for chi-square with 2 dof: p = exp(-JB/2) * (1 + JB/2)
    p = math.exp(-JB/2.0) * (1.0 + JB/2.0)
    return JB, p, S, K

# ===============================
# Projections / contraction operator
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

    # occasional rebin (if possible)
    if REBIN_EVERY > 0 and (step_index+1) % REBIN_EVERY == 0:
        n = len(z)
        if n >= 6 and (n//2) >= 4:  # keep enough points to fit deg up to 3/4 sometimes
            R = pairwise_rebin_matrix(n)
            x, z, V = apply_linear(R, x, z, V)

    return x, z, V

# ===============================
# D) explicit even-part involution projection
# ===============================

def even_projection_matrix(x):
    """
    Build B_even that maps z -> z_even where z_even(x_i) = 0.5(z(x_i)+z(x_j))
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

# ===============================
# Run analysis
# ===============================

def analyze_dataset(table_id, cov_id, label):
    print(f"\n=== {label} ===")

    t = get_json(table_id)
    c = get_json(cov_id)

    xbins, x, y = extract_xy(t)
    Vy = extract_cov_total(c, xbins)

    # log coords
    lx = np.log(x)
    z  = np.log(y)
    Vz = cov_to_logspace(Vy, y)

    # output CSV
    csv_path = os.path.join(OUTDIR, f"{label}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter","n",
            "best_deg","best_bic","best_chi2dof",
            "JB","JB_p","skew","excess_kurt",
            "outlier_frac_gt3",
            "R_hi_deg4",
            "odd_even_ratio_raw_deg4","odd_even_ratio_evenproj_deg4"
        ])

        # store for summary plots
        iters = []
        best_deg_series = []
        best_bic_series = []
        Rhi_series = []
        odd_ratio_raw_series = []
        odd_ratio_even_series = []

        for it in range(N_ITERS+1):
            n = len(z)

            # Fit all degrees; choose best by BIC
            fits = {}
            for deg in DEGREES:
                k = deg+1
                dof = n-k
                if dof <= 0:
                    continue
                beta, resid, chi2, Vinv, Vreg = gls_fit(lx, z, Vz, deg)
                aic, bic = aic_bic(chi2, k, n)
                fits[deg] = (beta, resid, chi2, bic, Vreg, dof)

            if not fits:
                break

            best_deg = min(fits.keys(), key=lambda d: fits[d][3])
            beta_b, resid_b, chi2_b, bic_b, Vreg_b, dof_b = fits[best_deg]
            chi2dof = chi2_b / dof_b

            # B: whitened residual normality
            W = cholesky_whitener(Vreg_b)
            wres = W @ resid_b
            JB, p, skew, exkurt = jarque_bera(wres)
            out_frac = float(np.mean(np.abs(wres) > 3.0))

            # C: higher-order "irrelevance" ratio from deg=4 (if fit exists)
            if 4 in fits:
                beta4 = fits[4][0]
                num = abs(beta4[3]) + abs(beta4[4])
                den = abs(beta4[0]) + abs(beta4[1]) + abs(beta4[2]) + 1e-12
                R_hi = float(num/den)
            else:
                R_hi = float("nan")

            # D: odd/even ratio (deg=4) before and after even-projection
            def odd_even_ratio(beta):
                odd = math.sqrt(beta[1]**2 + beta[3]**2)
                even = math.sqrt(beta[0]**2 + beta[2]**2 + beta[4]**2) + 1e-12
                return float(odd/even)

            odd_raw = float("nan")
            odd_even = float("nan")

            if 4 in fits:
                beta4_raw = fits[4][0]
                odd_raw = odd_even_ratio(beta4_raw)

                xc, B = even_projection_matrix(lx)
                zE = B @ z
                VE = B @ Vz @ B.T
                # fit in centered coord after projection
                beta4_E, resid_E, chi2_E, Vinv_E, Vreg_E = gls_fit(xc, zE, VE, 4)
                odd_even = odd_even_ratio(beta4_E)

            # write row
            writer.writerow([
                it, n,
                best_deg, bic_b, chi2dof,
                JB, p, skew, exkurt,
                out_frac,
                R_hi,
                odd_raw, odd_even
            ])

            # print concise metrics
            print(f"iter={it:2d} n={n:2d}  best_deg={best_deg}  chi2/dof={chi2dof:8.3f}  BIC={bic_b:10.3f}  "
                  f"JB={JB:7.3f} p≈{p:6.3g}  skew={skew:6.3f} exK={exkurt:6.3f}  |w|>3={out_frac:5.2%}  "
                  f"R_hi(deg4)={R_hi:7.3g}  odd/even raw→evenproj={odd_raw:7.3g}→{odd_even:7.3g}")

            # store for plots
            iters.append(it)
            best_deg_series.append(best_deg)
            best_bic_series.append(bic_b)
            Rhi_series.append(R_hi)
            odd_ratio_raw_series.append(odd_raw)
            odd_ratio_even_series.append(odd_even)

            # apply contraction for next step
            if it < N_ITERS:
                lx, z, Vz = contraction_step(lx, z, Vz, it)

        # summary plots
        def save_series_plot(y, title, ylabel, fname):
            plt.figure(figsize=(10,4))
            plt.plot(iters, y, marker='o')
            plt.title(f"{label}: {title}")
            plt.xlabel("iteration")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, fname), dpi=160)
            plt.close()

        save_series_plot(best_deg_series, "best degree (BIC)", "best_deg", f"{label}_bestdeg.png")
        save_series_plot(best_bic_series, "best BIC", "BIC", f"{label}_bestbic.png")
        save_series_plot(Rhi_series, "R_hi = (|b3|+|b4|)/(|b0|+|b1|+|b2|)", "R_hi", f"{label}_Rhi.png")

        # odd/even ratios
        plt.figure(figsize=(10,4))
        plt.plot(iters, odd_ratio_raw_series, marker='o', label="raw fit deg=4")
        plt.plot(iters, odd_ratio_even_series, marker='o', label="even-projected then fit deg=4")
        plt.title(f"{label}: odd/even ratio (deg=4)")
        plt.xlabel("iteration")
        plt.ylabel("odd/even")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{label}_odd_even.png"), dpi=160)
        plt.close()

    print(f"Saved metrics: {csv_path}")

def main():
    print(f"Output dir: {OUTDIR}")
    for table_id, cov_id, label in TABLES:
        analyze_dataset(table_id, cov_id, label)
    print("Done.")

if __name__ == "__main__":
    main()
