import math
import numpy as np
import requests
import matplotlib.pyplot as plt

TABLE_ID = 129890
COV_ID   = 129891
LABEL    = "pTll_50_76"

BASE_URL = "https://www.hepdata.net/record/{}?format=json"

def get_json(record_id):
    r = requests.get(BASE_URL.format(record_id), timeout=60)
    r.raise_for_status()
    return r.json()

def extract_xy(table_json):
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
    return xbins, np.array(xmid), np.array(y)

def extract_cov(cov_json, xbins):
    headers = [h["name"] for h in cov_json["headers"]]
    total_idx = [i for i,n in enumerate(headers) if "Total uncertainty" in n][0]
    y_col = total_idx - 2

    n = len(xbins)
    V = np.zeros((n,n))
    for row in cov_json["values"]:
        bi = (float(row["x"][0]["low"]), float(row["x"][0]["high"]))
        bj = (float(row["x"][1]["low"]), float(row["x"][1]["high"]))
        i = xbins.index(bi)
        j = xbins.index(bj)
        V[i,j] = float(row["y"][y_col]["value"])
        V[j,i] = V[i,j]
    return V

def design(x, deg):
    return np.vstack([x**k for k in range(deg+1)]).T

def fit_model(x, y, V, deg):
    X = design(x, deg)
    Vinv = np.linalg.inv(V)
    XtV = X.T @ Vinv
    beta = np.linalg.solve(XtV @ X, XtV @ y)
    resid = y - X @ beta
    chi2 = resid.T @ Vinv @ resid
    return beta, resid, chi2

# -------------------------

t = get_json(TABLE_ID)
c = get_json(COV_ID)

xbins, x, y = extract_xy(t)
Vy = extract_cov(c, xbins)

lx = np.log(x)
ly = np.log(y)
Vlog = Vy / np.outer(y,y)

degrees = [1,2,3,4]
fits = {}

for d in degrees:
    beta, resid, chi2 = fit_model(lx, ly, Vlog, d)
    fits[d] = (beta, resid, chi2)

# -------------------------
# Plot fits
# -------------------------

plt.figure(figsize=(12,8))

# Data
plt.errorbar(lx, ly, yerr=np.sqrt(np.diag(Vlog)),
             fmt='o', label='Data')

# Curves
x_dense = np.linspace(min(lx), max(lx), 400)

for d in degrees:
    beta = fits[d][0]
    Xd = design(x_dense, d)
    yd = Xd @ beta
    plt.plot(x_dense, yd, label=f"deg={d}")

plt.title(f"{LABEL} (log-log)")
plt.xlabel("log x")
plt.ylabel("log y")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# BIC bar chart
# -------------------------

bics = []
for d in degrees:
    chi2 = fits[d][2]
    k = d+1
    n = len(lx)
    bic = chi2 + k*np.log(n)
    bics.append(bic)

plt.figure()
plt.bar([str(d) for d in degrees], bics)
plt.title("BIC Comparison")
plt.xlabel("Polynomial degree")
plt.ylabel("BIC")
plt.show()

# -------------------------
# Residual heat
# -------------------------

plt.figure(figsize=(10,5))
for d in degrees:
    resid = fits[d][1]
    plt.plot(resid, label=f"deg={d}")
plt.title("Residuals")
plt.legend()
plt.show()
