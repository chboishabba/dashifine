import numpy as np
import json
import glob
import os
from numpy.linalg import eigvals, solve
from scipy.linalg import solve_discrete_lyapunov

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

ROOT = "hepdata_dashi_native"
OUT_DIR = "operator_definitive_cert"
BOOTSTRAPS = 100
MDL_LAMBDA = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load β-trajectory data
# ------------------------------------------------------------

def load_beta_states(root):
    """
    Expects per-iteration beta coefficient CSV or numpy files.
    Modify this loader if needed to match your structure.
    """
    states = []

    files = sorted(glob.glob(os.path.join(root, "**/*.csv"), recursive=True))
    for f in files:
        data = np.genfromtxt(f, delimiter=",", names=True)
        # assume columns b0,b1,b2,b3,b4 exist
        vec = np.array([data['b0'], data['b1'], data['b2'],
                        data['b3'], data['b4']]).flatten()
        states.append(vec)

    return np.array(states)


states = load_beta_states(ROOT)

if len(states) < 2:
    raise RuntimeError("Not enough β-iteration states found.")

X = states[:-1]
Y = states[1:]

# ------------------------------------------------------------
# 2. Estimate Global Jacobian
# ------------------------------------------------------------

# Least squares fit: Y ≈ X J^T
J = np.linalg.lstsq(X, Y, rcond=None)[0].T

np.save(os.path.join(OUT_DIR, "J_global.npy"), J)

# ------------------------------------------------------------
# 3. Contraction test
# ------------------------------------------------------------

eigvals_J = eigvals(J)
spectral_radius = max(abs(eigvals_J))

# contraction fraction
dist_in = np.linalg.norm(X[1:] - X[:-1], axis=1)
dist_out = np.linalg.norm(Y[1:] - Y[:-1], axis=1)
contraction_fraction = np.mean(dist_out < dist_in)

# ------------------------------------------------------------
# 4. Quadratic Invariant (Discrete Lyapunov)
# ------------------------------------------------------------

try:
    G = solve_discrete_lyapunov(J.T, np.eye(J.shape[0]))
    np.save(os.path.join(OUT_DIR, "G_invariant.npy"), G)
except Exception as e:
    G = None
    print("Lyapunov solve failed:", e)

# ------------------------------------------------------------
# 5. Signature extraction
# ------------------------------------------------------------

def signature(matrix):
    eig = np.linalg.eigvalsh(matrix)
    pos = np.sum(eig > 1e-8)
    neg = np.sum(eig < -1e-8)
    zero = len(eig) - pos - neg
    return int(pos), int(neg), int(zero)

if G is not None:
    sig = signature(G)
else:
    sig = None

# ------------------------------------------------------------
# 6. Bootstrap stability
# ------------------------------------------------------------

bootstrap_sigs = []

for _ in range(BOOTSTRAPS):
    idx = np.random.choice(len(X), len(X), replace=True)
    Xb = X[idx]
    Yb = Y[idx]
    Jb = np.linalg.lstsq(Xb, Yb, rcond=None)[0].T
    try:
        Gb = solve_discrete_lyapunov(Jb.T, np.eye(Jb.shape[0]))
        bootstrap_sigs.append(signature(Gb))
    except:
        pass

# ------------------------------------------------------------
# 7. MDL proxy descent
# ------------------------------------------------------------

def mdl_proxy(state):
    model_len = np.linalg.norm(state)
    residual_len = np.var(state)
    return model_len + MDL_LAMBDA * residual_len

mdl_values = np.array([mdl_proxy(s) for s in states])
mdl_decreasing = np.all(np.diff(mdl_values) <= 1e-8)

# ------------------------------------------------------------
# 8. Save Report
# ------------------------------------------------------------

report = {
    "spectral_radius": float(spectral_radius),
    "contraction_fraction": float(contraction_fraction),
    "signature": sig,
    "bootstrap_signatures": bootstrap_sigs[:10],
    "mdl_monotone_descent": bool(mdl_decreasing),
}

with open(os.path.join(OUT_DIR, "definitive_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\n=== Definitive Certification Report ===")
print(json.dumps(report, indent=2))
