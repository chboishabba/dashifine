import numpy as np

from dashifine.kernels import dashifine_kernel


def make_wave_field(n_points=1500, M=4, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n_points, 2))

    ks = rng.integers(-4, 5, size=(M, 2))
    ks = ks[(ks[:, 0] != 0) | (ks[:, 1] != 0)]
    while len(ks) < M:
        k_new = rng.integers(-4, 5, size=(1, 2))
        if (k_new[0, 0] != 0) or (k_new[0, 1] != 0):
            ks = np.vstack([ks, k_new])
    ks = ks[:M]

    A = rng.normal(0.0, 1.0, size=(M,))
    phi = rng.uniform(0.0, 2 * np.pi, size=(M,))

    y = np.zeros((n_points,), dtype=float)
    for m in range(M):
        y += A[m] * np.sin(X @ ks[m] + phi[m])

    y += noise * rng.normal(size=y.shape)
    return X, y


def krr_predict(Xtr, ytr, Xte, kernel_fn, lam=1e-3):
    K = kernel_fn(Xtr, Xtr)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), ytr)
    return kernel_fn(Xte, Xtr) @ alpha


def rbf_kernel(X1, X2, lengthscale=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    d2 = X1_sq + X2_sq - 2 * (X1 @ X2.T)
    return np.exp(-0.5 * d2 / (lengthscale**2))


def periodic_rbf_kernel(X1, X2, lengthscale=1.0, period=2 * np.pi):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    diff = np.sin(diff * np.pi / period) * (period / np.pi)
    d2 = np.sum(diff**2, axis=-1)
    return np.exp(-0.5 * d2 / (lengthscale**2))


def kernel_spectrum(X, kernel_fn, lam=1e-12):
    K = kernel_fn(X, X)
    K = K + lam * np.eye(len(X))
    eigvals = np.linalg.eigvalsh(K)
    return eigvals[::-1]


def mse(a, b):
    return float(np.mean((a - b) ** 2))


if __name__ == "__main__":
    X, y = make_wave_field(n_points=2000, M=5, noise=0.03, seed=42)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_train = 200
    tr, te = idx[:n_train], idx[n_train:]

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    print("\nKernel eigenspectra (top 10 eigenvalues):")
    for T in [0.3, 0.7, 1.2, 2.0, 4.0]:
        eigs = kernel_spectrum(
            Xtr,
            kernel_fn=lambda A, B, T=T: dashifine_kernel(
                A,
                B,
                k_max=4,
                temperature=T,
            ),
        )
        print(f"T={T:>4}  eigs[:10]={np.round(eigs[:10], 4)}")

    print("\nRBF eigenspectra (top 10 eigenvalues):")
    for ell in [0.3, 0.7, 1.2, 2.0, 4.0]:
        eigs = kernel_spectrum(
            Xtr,
            kernel_fn=lambda A, B, ell=ell: rbf_kernel(
                A,
                B,
                lengthscale=ell,
            ),
        )
        print(f"ell={ell:>4}  eigs[:10]={np.round(eigs[:10], 4)}")

    print("\nPeriodic RBF eigenspectra (top 10 eigenvalues):")
    for ell in [0.3, 0.7, 1.2, 2.0, 4.0]:
        eigs = kernel_spectrum(
            Xtr,
            kernel_fn=lambda A, B, ell=ell: periodic_rbf_kernel(
                A,
                B,
                lengthscale=ell,
            ),
        )
        print(f"ell={ell:>4}  eigs[:10]={np.round(eigs[:10], 4)}")

    for T in [0.3, 0.7, 1.2, 2.0, 4.0]:
        yhat = krr_predict(
            Xtr,
            ytr,
            Xte,
            kernel_fn=lambda A, B, T=T: dashifine_kernel(
                A,
                B,
                k_max=4,
                temperature=T,
            ),
            lam=1e-2,
        )
        print(f"T={T:>4}  test MSE={mse(yhat, yte):.6f}")

    print("\nRBF baselines:")
    for ell in [0.3, 0.7, 1.2, 2.0, 4.0]:
        yhat = krr_predict(
            Xtr,
            ytr,
            Xte,
            kernel_fn=lambda A, B, ell=ell: rbf_kernel(
                A,
                B,
                lengthscale=ell,
            ),
            lam=1e-2,
        )
        print(f"RBF ell={ell:>4}  test MSE={mse(yhat, yte):.6f}")

    print("\nPeriodic RBF baselines:")
    for ell in [0.3, 0.7, 1.2, 2.0, 4.0]:
        yhat = krr_predict(
            Xtr,
            ytr,
            Xte,
            kernel_fn=lambda A, B, ell=ell: periodic_rbf_kernel(
                A,
                B,
                lengthscale=ell,
            ),
            lam=1e-2,
        )
        print(f"pRBF ell={ell:>4}  test MSE={mse(yhat, yte):.6f}")
