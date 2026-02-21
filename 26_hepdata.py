import math
import numpy as np
import requests

# --- choose dataset (this is the small 12-bin one) ---
T11 = "https://www.hepdata.net/record/129890?format=json"  # pT ll mass 50-76 jet
T12 = "https://www.hepdata.net/record/129891?format=json"  # covariance for that table

def get_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def extract_xy(table_json: dict):
    # Independent var is bins with low/high; dependent is y with "value"
    vals = table_json["values"]
    lows, highs, ys = [], [], []
    for row in vals:
        xbin = row["x"][0]
        yval = row["y"][0]["value"]
        lows.append(float(xbin["low"]))
        highs.append(float(xbin["high"]))
        ys.append(float(yval))
    lows = np.array(lows)
    highs = np.array(highs)
    xmid = 0.5 * (lows + highs)
    # avoid log(0) if first bin starts at 0
    xmid = np.where(xmid <= 0, 0.5 * highs, xmid)
    y = np.array(ys)
    return xmid, y

def extract_cov(cov_json: dict, x_bins):
    """
    HEPData covariance tables are often provided as a "long" table:
    (bin_i, bin_j) -> cov_ij, with several covariance components.
    We’ll use the "Total uncertainty" column.
    """
    headers = [h["name"] for h in cov_json["headers"]]
    # Find the "Total uncertainty" column (name varies a bit, so do a contains match)
    total_idx = None
    for i, name in enumerate(headers):
        if "Total uncertainty" in name:
            total_idx = i
            break
    if total_idx is None:
        raise ValueError(f"Could not find Total uncertainty column. headers={headers}")

    n = len(x_bins)
    V = np.zeros((n, n), dtype=float)

    # Each row gives x (bin_i), x__1 (bin_j) plus y entries that include covariance values.
    for row in cov_json["values"]:
        xi = row["x"][0]
        xj = row["x"][1]  # second independent variable is usually at index 1
        # Map (low,high) bins to indices
        bi = (float(xi["low"]), float(xi["high"]))
        bj = (float(xj["low"]), float(xj["high"]))
        i = x_bins.index(bi)
        j = x_bins.index(bj)

        cov_ij = float(row["y"][0]["value"]) if total_idx == 2 else float(row["y"][total_idx - 2]["value"])
        # NOTE: depending on the exact JSON schema, y may be a list of columns.
        # If this fails, print row["y"] and adjust indexing to match your JSON.
        V[i, j] = cov_ij
        V[j, i] = cov_ij

    return V

def design_matrix(lx: np.ndarray, degree: int):
    # polynomial in log(x): [1, lx, lx^2, ...]
    return np.vstack([lx**k for k in range(degree + 1)]).T

def gls_fit(X, y, Vinv):
    # beta = (X^T V^-1 X)^-1 X^T V^-1 y
    XtV = X.T @ Vinv
    M = XtV @ X
    b = np.linalg.solve(M, XtV @ y)
    resid = y - X @ b
    chi2 = float(resid.T @ Vinv @ resid)
    return b, resid, chi2

def mdl_score(chi2: float, k: int, n: int):
    # A simple MDL-like score: chi2 + k*log(n)
    # (If you want your exact DASHI MDL, swap it here.)
    return chi2 + k * math.log(n)

def main():
    t11 = get_json(T11)
    x, y = extract_xy(t11)

    # log-space test (common for falling spectra)
    lx = np.log(x)
    ly = np.log(y)

    # If you can load covariance:
    t12 = get_json(T12)

    # Build the bin list so we can index covariance by (low,high)
    x_bins = [(float(row["x"][0]["low"]), float(row["x"][0]["high"])) for row in t11["values"]]
    V = extract_cov(t12, x_bins)

    # regularize in case of near-singularity
    eps = 1e-18 * np.trace(V) / len(V)
    Vinv = np.linalg.inv(V + eps * np.eye(len(V)))

    results = []
    for deg in [1, 2, 3, 4]:
        X = design_matrix(lx, deg)
        k = deg + 1
        beta, resid, chi2 = gls_fit(X, ly, Vinv)
        score = mdl_score(chi2, k, len(lx))
        results.append((deg, score, chi2, beta))

    results.sort(key=lambda t: t[1])
    for deg, score, chi2, beta in results:
        print(f"deg={deg}  MDL~={score:.3f}  chi2={chi2:.3f}  beta={beta}")

if __name__ == "__main__":
    main()
