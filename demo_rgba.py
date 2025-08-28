"""Generate 2‑D RGBA slices of a synthetic 4‑D CMYK field.

This module provides a :func:`cmyk_slice_rgba` helper that evaluates the
four‑class field (cyan, magenta, yellow, black) on a 2‑D slice of 4‑D space
and returns an RGBA image where the alpha channel encodes overall field
strength.  A demo at the bottom of the file now saves two sets of 360 frames:

1. A rotation around the unit circle in the ``(z, w)`` plane.
2. A top‑to‑bottom scan that sweeps ``w`` from +1 to −1 while ``z`` stays fixed.
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


# ------------------------------
# Core utilities
# ------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def _class_field_xy(
    X: np.ndarray,
    Y: np.ndarray,
    z0: float,
    w0: float,
    centers: Iterable[np.ndarray],
    sharpness: float,
) -> np.ndarray:
    """Evaluate the field for one class at fixed ``(z0, w0)``."""
    f = np.zeros_like(X, dtype=np.float32)
    for c in centers:
        dx = X - c[0]
        dy = Y - c[1]
        dz = z0 - c[2]
        dw = w0 - c[3]
        d = np.sqrt(dx * dx + dy * dy + dz * dz + dw * dw, dtype=np.float32)
        f += gelu((1.0 - d) * sharpness).astype(np.float32)
    return f


def cmyk_slice_rgba(
    z0: float,
    w0: float,
    res: int = 360,
    sharpness: float = 2.8,
    tie_gamma: float = 0.9,
    penalty_strength: float = 0.35,
    alpha_threshold: float = 0.05,
) -> np.ndarray:
    """Return an RGBA image for a slice of the CMYK field.

    The alpha channel represents overall field strength and is thresholded to
    zero below ``alpha_threshold`` to distinguish empty regions from dark
    colours.
    """
    x = np.linspace(-1, 1, res, dtype=np.float32)
    y = np.linspace(-1, 1, res, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 4‑D centres for CMYK classes
    C_centers = [np.array([0.5, 0.4, -0.2, 0.3], dtype=np.float32)]
    M_centers = [np.array([-0.6, 0.1, 0.6, -0.4], dtype=np.float32)]
    Y_centers = [np.array([0.1, -0.5, -0.4, 0.5], dtype=np.float32)]
    K_centers = [np.array([-0.2, -0.3, 0.5, -0.6], dtype=np.float32)]

    C = _class_field_xy(X, Y, z0, w0, C_centers, sharpness)
    M = _class_field_xy(X, Y, z0, w0, M_centers, sharpness)
    Yf = _class_field_xy(X, Y, z0, w0, Y_centers, sharpness)
    K = _class_field_xy(X, Y, z0, w0, K_centers, sharpness)

    stack = np.stack([C, M, Yf, K], axis=0)
    max1 = np.max(stack, axis=0)
    mask = stack == max1[None, ...]
    stack_masked = np.where(mask, -np.inf, stack)
    max2 = np.max(stack_masked, axis=0)

    tie_pen = gelu(-tie_gamma * (max1 - max2).astype(np.float32)).astype(np.float32)
    for f in (C, M, Yf, K):
        f *= (1 - penalty_strength * tie_pen)

    eps = 1e-7
    S = C + M + Yf + K + eps
    wC, wM, wY, wK = C / S, M / S, Yf / S, K / S

    R = (1 - wM) * (1 - wK)
    G = (1 - wY) * (1 - wK)
    B = (1 - wC) * (1 - wK)

    alpha = np.clip(S / S.max(), 0, 1).astype(np.float32)
    alpha = np.where(alpha < alpha_threshold, 0.0, alpha)

    return np.clip(np.stack([R, G, B, alpha], axis=-1), 0, 1).astype(np.float32)


# ------------------------------
# Demo: 360‑frame rotation and vertical scan
# ------------------------------

def main() -> None:
    out_dir_rot = "cmyk_rgba_rot"
    out_dir_scan = "cmyk_rgba_scan"
    os.makedirs(out_dir_rot, exist_ok=True)
    os.makedirs(out_dir_scan, exist_ok=True)

    # 1) rotation around unit circle in z–w plane
    for i in range(360):
        theta = np.deg2rad(i)
        z0 = np.cos(theta)
        w0 = np.sin(theta)
        img = cmyk_slice_rgba(z0, w0)
        plt.imsave(f"{out_dir_rot}/slice_{i:03d}.png", img)

    # 2) top-to-bottom scan: sweep w from +1 to -1 at fixed z=0
    for i in range(360):
        w0 = 1 - 2 * (i / 359)
        z0 = 0.0
        img = cmyk_slice_rgba(z0, w0)
        plt.imsave(f"{out_dir_scan}/slice_{i:03d}.png", img)

    print(
        "360 RGBA slices saved to '{}' and '{}'".format(
            f"{out_dir_rot}/slice_###.png", f"{out_dir_scan}/slice_###.png"
        )
    )


if __name__ == "__main__":
    main()
