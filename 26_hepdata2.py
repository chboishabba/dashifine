import math
import numpy as np
import requests

# ===============================
# CONFIG: CMS DY + ≥1 jet tables
# (your same IDs)
# ===============================
TABLES = [
    (129890, 129891, "pTll_50_76"),
    (129892, 129893, "pTll_76_106"),
    (129894, 129895, "pTll_106_170"),
    (129896, 129897, "pTll_170_350"),
    (129902, 129903, "phistar_50_76"),
]

BASE_URL = "https://www.hepdata.net/record/{}?format=json"

# ===============================
# Download / parse
# ===============================

def get_json(record_id: int) -> dict:
    url = BASE_URL.format(record_id)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def extract_xy(table_json: dict):
    vals = table_json["values"]
    xbins = []
    xmid = []
    y = []

    for row in vals:
        xb = row["x"][0]
        low = float(xb["low"])
        high = float(xb["high"])
        mid = 0.5 * (low + high)
        if mid <= 0:
            mid = 0.5 * high
        xbins.append((low, high))
        xmid.append(mid)
        y.append(float(row["y"][0]["value"]))

    return xbins, np.array(xmid, float), np.array(y, float)

def extract_cov_total(cov_json: dict, xbins):
    """
    Build full covariance matrix V (y-space) from HEPData covariance table.
    Assumes the covariance table has two independent variables (bin_i, bin_j)
    and y-columns including "Total uncertainty".
    """
    headers = [h["name"] for h in cov_json["headers"]]

    total_idx = None
    for i, name in enumerate(headers):
        if "Total uncertainty" in name:
            total_idx = i
            break
    if total_idx is None:
        raise ValueError(f"Could not find 'Total uncertainty' in headers: {headers}")

    n = len(xbins)
    V = np.zeros((n, n), float)

    # y columns correspond to headers AFTER the independent variables.
    # In the JSON schema used here, independent variables occupy two 'x' entries,
    # and y is a list aligned to dependent variable headers.
    # We assume: y[col_index - num_indep_headers] mapping; for these records it's (total_idx - 2).
    y_col = total_idx - 2
    if y_col < 0:
        raise ValueError("Unexpected header layout; cannot map total covariance column.")

    for row in cov_json["values"]:
        xi = row["x"][0]
        xj = row["x"][1]
        bi = (float(xi["low"]), float(xi["high"]))
        bj = (float(xj["low"]), float(xj["high"]))

        try:
            i = xbins.index(bi)
            j = xbins.index(bj)
        except ValueError:
            raise ValueError(f"Bin pair not found in xbins: {bi}, {bj}")

        cov_ij = float(row["y"][y_col]["value"])
        V[i, j] = cov_ij
        V[j, i] = cov_ij

    return V

# ===============================
# Model fitting
# ===============================

def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vstack([x**k for k in range(degree + 1)]).T

def gls_fit(X: np.ndarray, y: np.ndarray, V: np.ndarray, ridge: float = 0.0):
    """
    GLS: beta = (X^T V^-1 X)^-1 X^T V^-1 y
    Uses a small ridge on V if needed.
    """
    n = len(y)
    if ridge > 0:
        V = V + ridge * np.eye(n)

    Vinv = np.linalg.inv(V)
    XtV = X.T @ Vinv
    M = XtV @ X
    beta = np.linalg.solve(M, XtV @ y)
    resid = y - X @ beta
    chi2 = float(resid.T @ Vinv @ resid)
    return beta, chi2

def aic_bic(chi2: float, k: int, n: int):
    aic = chi2 + 2 * k
    bic = chi2 + k * math.log(n)
    return aic, bic

# ===============================
# Covariance transforms
# ===============================

def cov_to_logspace(Vy: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    If z = log(y), then Cov(z) ~ J Cov(y) J where J = diag(1/y).
    """
    return Vy / np.outer(y, y)

# ===============================
# Projections (linear maps)
# ===============================

def pairwise_rebin_matrix(n: int) -> np.ndarray:
    """
    Build A such that y' = A y merges bins (0,1), (2,3), ...
    We take simple average (0.5,0.5). You can swap to inverse-var weights later.
    """
    m = n // 2
    A = np.zeros((m, n), float)
    for r in range(m):
        A[r, 2*r] = 0.5
        A[r, 2*r + 1] = 0.5
    return A

def smooth3_matrix(n: int) -> np.ndarray:
    """
    3-pt moving average with edge points unchanged:
      y'[0]=y[0], y'[n-1]=y[n-1],
      y'[i]=(y[i-1]+y[i]+y[i+1])/3
    """
    A = np.zeros((n, n), float)
    A[0, 0] = 1.0
    A[n-1, n-1] = 1.0
    for i in range(1, n-1):
        A[i, i-1] = 1/3
        A[i, i]   = 1/3
        A[i, i+1] = 1/3
    return A

def apply_projection(A: np.ndarray, x: np.ndarray, y: np.ndarray, V: np.ndarray):
    """
    y' = A y
    V' = A V A^T
    For x: take the same linear map on x as a rough proxy (works for monotone bins).
    """
    y2 = A @ y
    V2 = A @ V @ A.T
    x2 = A @ x
    return x2, y2, V2

# ===============================
# Main analysis
# ===============================

def run_models(tag, x, y, V, degrees=(1,2,3,4), ridge_scale=1e-12):
    n = len(y)
    # Regularize V to avoid singularities:
    ridge = ridge_scale * np.trace(V) / max(n,1)
    results = []
    for deg in degrees:
        k = deg + 1
        dof = n - k
        if dof <= 0:
            continue
        X = design_matrix(x, deg)
        beta, chi2 = gls_fit(X, y, V, ridge=ridge)
        aic, bic = aic_bic(chi2, k, n)
        results.append((deg, chi2, chi2/dof, aic, bic, beta))
    results.sort(key=lambda t: t[4])  # sort by BIC
    print(f"\n--- {tag} (n={n}) ---")
    for deg, chi2, chi2dof, aic, bic, beta in results:
        print(f"deg={deg}  chi2={chi2:.3f}  chi2/dof={chi2dof:.3f}  AIC={aic:.3f}  BIC={bic:.3f}")
    return results

def analyze_dataset(table_id, cov_id, label, fit_in_logy=True, fit_in_logx=True):
    print("\n==============================")
    print(f"Dataset: {label}")
    print("==============================")

    t = get_json(table_id)
    c = get_json(cov_id)

    xbins, x, y = extract_xy(t)
    Vy = extract_cov_total(c, xbins)

    # Choose transform
    if fit_in_logx:
        x_fit = np.log(x)
    else:
        x_fit = x

    if fit_in_logy:
        y_fit = np.log(y)
        V_fit = cov_to_logspace(Vy, y)  # <-- critical fix
    else:
        y_fit = y
        V_fit = Vy

    # RAW
    run_models("RAW", x_fit, y_fit, V_fit)

    # PAIRWISE REBIN
    n = len(y_fit)
    A = pairwise_rebin_matrix(n)
    x2, y2, V2 = apply_projection(A, x_fit, y_fit, V_fit)
    run_models("PAIRWISE REBIN", x2, y2, V2)

    # 3-POINT SMOOTH
    S = smooth3_matrix(n)
    x3, y3, V3 = apply_projection(S, x_fit, y_fit, V_fit)
    run_models("3-POINT SMOOTH", x3, y3, V3)

if __name__ == "__main__":
    print("Fitting options: logx=True, logy=True (with covariance propagation for logy)")
    for table_id, cov_id, label in TABLES:
        analyze_dataset(table_id, cov_id, label, fit_in_logy=True, fit_in_logx=True)
