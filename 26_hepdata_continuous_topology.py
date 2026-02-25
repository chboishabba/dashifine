import numpy as np
import pandas as pd
from pathlib import Path
from ripser import ripser
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# ==========================================
# Utilities
# ==========================================

def max_persistence(dgm):
    if len(dgm) == 0:
        return 0.0
    return float(np.max(dgm[:, 1] - dgm[:, 0]))


def compute_ph(X, maxdim=2):
    dgms = ripser(X, maxdim=maxdim)["dgms"]
    H0 = max_persistence(dgms[0])
    H1 = max_persistence(dgms[1])
    H2 = max_persistence(dgms[2]) if len(dgms) > 2 else 0.0
    return H0, H1, H2, dgms


# ==========================================
# TwoNN Intrinsic Dimension Estimator
# (Facco et al. 2017)
# ==========================================

def twonn_dimension(X):
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)

    r1 = dists[:, 1]
    r2 = dists[:, 2]

    mu = r2 / (r1 + 1e-12)
    mu = mu[mu > 0]

    log_mu = np.log(mu)
    m = 1.0 / (np.mean(log_mu) + 1e-12)
    return float(m)


# ==========================================
# Load continuous lens field
# ==========================================

def load_continuous_lens_space(root="hepdata_to_dashi"):
    root = Path(root)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    all_rows = []

    for d in subdirs:
        f = d / "lenses_continuous.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        numeric = df.select_dtypes(include=[np.number]).columns
        if "bin" in numeric:
            numeric = numeric.drop("bin")
        X = df[numeric].values
        if X.shape[1] >= 10:
            all_rows.append(X[:, :10])

    if not all_rows:
        raise RuntimeError("No continuous lens data found.")

    return np.vstack(all_rows)


# ==========================================
# Gaussian Null with same covariance
# ==========================================

def gaussian_null(X):
    mu = np.mean(X, axis=0)
    cov = np.cov(X.T)
    return np.random.multivariate_normal(mu, cov, size=len(X))


# ==========================================
# Contraction trajectory generator
# (internal coefficient shrink model)
# ==========================================

def contraction_trajectory(X, n_steps=20):
    """
    Simple contraction: PCA compress toward first components
    """
    pca = PCA(n_components=X.shape[1])
    Z = pca.fit_transform(X)
    traj = []

    for t in np.linspace(0, 1, n_steps):
        Zt = Z * (1 - t)
        Xt = pca.inverse_transform(Zt)
        traj.append(Xt)

    return np.vstack(traj)


# ==========================================
# Residual subspace selection
# (top entropy lenses)
# ==========================================

def highest_entropy_subspace(X, k=4):
    ent = []
    for j in range(X.shape[1]):
        hist, _ = np.histogram(X[:, j], bins=20, density=True)
        hist = hist[hist > 0]
        ent.append(-np.sum(hist * np.log(hist)))

    ent = np.array(ent)
    idx = np.argsort(ent)[::-1][:k]
    return X[:, idx], idx


# ==========================================
# Main Analysis
# ==========================================

def main():

    print("\nLoading continuous 10D lens space...")
    X = load_continuous_lens_space()
    print("Total points:", X.shape[0])
    print("Dimension:", X.shape[1])

    # ----------------------------------
    # PH on continuous space
    # ----------------------------------
    print("\n=== PH on Continuous Lens Space ===")
    H0, H1, H2, _ = compute_ph(X)
    print("H1:", round(H1, 4))
    print("H2:", round(H2, 4))

    # ----------------------------------
    # Intrinsic dimension
    # ----------------------------------
    print("\n=== TwoNN Intrinsic Dimension ===")
    dim_est = twonn_dimension(X)
    print("Estimated dimension:", round(dim_est, 3))

    # ----------------------------------
    # Gaussian null
    # ----------------------------------
    print("\n=== Gaussian Null Comparison ===")
    Xnull = gaussian_null(X)
    H0n, H1n, H2n, _ = compute_ph(Xnull)
    dim_null = twonn_dimension(Xnull)

    print("Null H1:", round(H1n, 4))
    print("Null H2:", round(H2n, 4))
    print("Null dim:", round(dim_null, 3))

    # ----------------------------------
    # Residual subspace
    # ----------------------------------
    print("\n=== Residual Subspace PH ===")
    Xsub, idx = highest_entropy_subspace(X, k=4)
    H0s, H1s, H2s, _ = compute_ph(Xsub)
    print("Using lenses:", idx)
    print("Subspace H1:", round(H1s, 4))
    print("Subspace H2:", round(H2s, 4))

    # ----------------------------------
    # Contraction trajectory PH
    # ----------------------------------
    print("\n=== Contraction Trajectory PH ===")
    Xtraj = contraction_trajectory(X, n_steps=20)
    H0t, H1t, H2t, _ = compute_ph(Xtraj)
    print("Trajectory H1:", round(H1t, 4))
    print("Trajectory H2:", round(H2t, 4))


if __name__ == "__main__":
    main()
