import numpy as np

# ----------------------------
# 1) Synthetic "wave field" task
# ----------------------------
def make_wave_field(n_points=1500, M=4, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n_points, 2))  # (x1, x2)

    # Random integer wavevectors (small-ish frequencies)
    ks = rng.integers(low=-4, high=5, size=(M, 2))
    ks = ks[(ks[:, 0] != 0) | (ks[:, 1] != 0)]  # avoid (0,0)
    while len(ks) < M:
        k_new = rng.integers(low=-4, high=5, size=(1, 2))
        if (k_new[0, 0] != 0) or (k_new[0, 1] != 0):
            ks = np.vstack([ks, k_new])
    ks = ks[:M]

    A = rng.normal(0.0, 1.0, size=(M,))
    phi = rng.uniform(0.0, 2*np.pi, size=(M,))

    y = np.zeros((n_points,), dtype=float)
    for m in range(M):
        y += A[m] * np.sin(X @ ks[m] + phi[m])

    y += noise * rng.normal(size=y.shape)
    return X, y, ks, A, phi

# ----------------------------
# 2) A baseline kernel (RBF). Replace this with your dashifine kernel.
# ----------------------------
def rbf_kernel(X1, X2, lengthscale=1.0):
    # K_ij = exp(-||x_i - x_j||^2 / (2 l^2))
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    d2 = X1_sq + X2_sq - 2 * (X1 @ X2.T)
    return np.exp(-0.5 * d2 / (lengthscale**2))

# ----------------------------
# 3) Kernel ridge regression
# ----------------------------
def krr_fit_predict(X_train, y_train, X_test, kernel_fn, lam=1e-3):
    K = kernel_fn(X_train, X_train)
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y_train)
    K_test = kernel_fn(X_test, X_train)
    y_pred = K_test @ alpha
    return y_pred

def mse(a, b):
    return float(np.mean((a - b) ** 2))

if __name__ == "__main__":
    X, y, ks, A, phi = make_wave_field(n_points=2000, M=5, noise=0.03, seed=42)

    rng = np.random.default_rng(123)
    idx = rng.permutation(len(X))
    n_train = 200  # deliberately small: forces "generalization"
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    # Try a couple lengthscales to see if the kernel is capturing low-frequency structure
    for ell in [0.3, 0.7, 1.2, 2.0]:
        yhat = krr_fit_predict(
            Xtr, ytr, Xte,
            kernel_fn=lambda A, B, ell=ell: rbf_kernel(A, B, lengthscale=ell),
            lam=1e-2
        )
        print(f"ell={ell:>3}  test MSE={mse(yhat, yte):.5f}")
