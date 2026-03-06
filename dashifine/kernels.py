import numpy as np


def dashifine_kernel(
    X1,
    X2,
    *,
    k_max=4,
    temperature=1.0,
    eps=1e-8,
):
    """
    Dashifine spectral kernel for R^2 inputs.

    K(x, x') = sum_{k in Z^2, |k|<=k_max} exp(-||k||^2 / T) * cos(<k, x - x'>)

    This kernel is stationary and PSD via Bochner's theorem.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    ks = []
    for k1 in range(-k_max, k_max + 1):
        for k2 in range(-k_max, k_max + 1):
            if k1 == 0 and k2 == 0:
                continue
            ks.append((k1, k2))
    ks = np.asarray(ks, dtype=float)

    k_norm2 = np.sum(ks**2, axis=1)
    w = np.exp(-k_norm2 / max(float(temperature), eps))

    diff = X1[:, None, :] - X2[None, :, :]
    phase = diff @ ks.T
    K = np.sum(w[None, None, :] * np.cos(phase), axis=-1)

    K /= (np.sum(w) + eps)
    return K
