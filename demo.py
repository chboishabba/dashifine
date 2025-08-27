"""Demonstration script for generating and saving 4D colour field slices.

This module performs a coarse int8 search to find an interesting 2D slice of a
synthetic 4D field, refines the slice parameters in float32 space, and then
renders a series of rotated slices. The resulting images and a summary JSON
are saved to ``/mnt/data``.

The implementation mirrors the demo provided in the project description and is
intended to serve as a minimal, dependency‑free example.  It does not depend on
the package's ``Main_with_rotation`` module and can be run directly:

```
python demo.py
```

Running the script will create PNG images and a ``summary.json`` file in the
``/mnt/data`` directory and print a JSON summary to stdout.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


# ---------------- Configurable parameters ----------------
RES_HI = 420
RES_COARSE = 56
SHARPNESS = 2.8
TIE_GAMMA = 0.9
TIE_STRENGTH = 0.35
INTENSITY_SCALE = True

Z0_RANGE = (-0.4, 0.4)
Z0_STEPS = 5
W0_RANGE = (-0.4, 0.4)
W0_STEPS = 5
SLOPES = np.array([-0.4, 0.0, 0.4], dtype=np.float32)

ROT_BASE_DEG = 18.0  # evenly cover ~180 degrees with 10 slices
NUM_ROTATED = 10  # produce 10 slices
SEED = 7

# Define 4D class centres (C, M, Y, K)
C_centers = [np.array([0.5, 0.4, -0.2, 0.3], dtype=np.float32)]
M_centers = [np.array([-0.6, 0.1, 0.6, -0.4], dtype=np.float32)]
Y_centers = [np.array([0.1, -0.5, -0.4, 0.5], dtype=np.float32)]
K_centers = [np.array([-0.2, -0.3, 0.5, -0.6], dtype=np.float32)]
CLASSES = [C_centers, M_centers, Y_centers, K_centers]


# ---------------- Helper functions ----------------
def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def eval_slice_affine(
    res: int,
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    sharpness: float = SHARPNESS,
    tie_gamma: float = TIE_GAMMA,
    tie_strength: float = TIE_STRENGTH,
    intensity_scale: bool = INTENSITY_SCALE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the colour field on a 2D slice defined by origin ``o`` and axes
    ``a`` and ``b``.

    Returns an RGB image, raw class fields and the sum of fields.
    """

    u = np.linspace(-1, 1, res, dtype=np.float32)
    v = np.linspace(-1, 1, res, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")

    X = o[0] + U * a[0] + V * b[0]
    Y = o[1] + U * a[1] + V * b[1]
    Z = o[2] + U * a[2] + V * b[2]
    W = o[3] + U * a[3] + V * b[3]

    fields: List[np.ndarray] = []
    for centers in CLASSES:
        f = np.zeros_like(X, dtype=np.float32)
        for c in centers:
            dx = X - c[0]
            dy = Y - c[1]
            dz = Z - c[2]
            dw = W - c[3]
            d = np.sqrt(dx * dx + dy * dy + dz * dz + dw * dw, dtype=np.float32)
            f += gelu((1.0 - d) * sharpness).astype(np.float32)
        fields.append(f)

    fields = np.stack(fields, axis=0)
    max1 = np.max(fields, axis=0)
    neg_inf = np.full_like(fields, -np.inf, dtype=np.float32)
    mask = fields == max1[None, ...]
    fields_masked = np.where(mask, neg_inf, fields)
    max2 = np.max(fields_masked, axis=0)
    tie_pen = gelu(-tie_gamma * (max1 - max2).astype(np.float32)).astype(np.float32)
    fields *= (1 - tie_strength * tie_pen)[None, ...]

    S = fields.sum(axis=0) + np.float32(1e-7)
    w = fields / S[None, ...]
    wC, wM, wY, wK = w[0], w[1], w[2], w[3]
    R = (1 - wM) * (1 - wK)
    G = (1 - wY) * (1 - wK)
    B = (1 - wC) * (1 - wK)

    if intensity_scale:
        intensity = np.clip(S / S.max(), 0, 1).astype(np.float32)
        R *= intensity
        G *= intensity
        B *= intensity

    RGB = np.clip(np.stack([R, G, B], axis=-1), 0, 1).astype(np.float32)
    return RGB, fields, S


def score_float32(RGB: np.ndarray, S: np.ndarray) -> float:
    act = float(S.mean())
    var = float(np.var(RGB.reshape(-1, 3), axis=0).mean())
    return 0.6 * act + 0.4 * var


def coarse_int8_search(res: int = RES_COARSE):
    z0_vals = np.linspace(Z0_RANGE[0], Z0_RANGE[1], Z0_STEPS, dtype=np.float32)
    w0_vals = np.linspace(W0_RANGE[0], W0_RANGE[1], W0_STEPS, dtype=np.float32)
    slopes = SLOPES
    best = None
    best_params = None
    for z0 in z0_vals:
        for w0 in w0_vals:
            for sz_u in slopes:
                for sw_u in slopes:
                    for sz_v in slopes:
                        for sw_v in slopes:
                            o = np.array([0.0, 0.0, z0, w0], dtype=np.float32)
                            a = np.array([1.0, 0.0, sz_u, sw_u], dtype=np.float32)
                            b = np.array([0.0, 1.0, sz_v, sw_v], dtype=np.float32)
                            RGB, fields, S = eval_slice_affine(res, o, a, b)
                            fmin, fmax = fields.min(), fields.max()
                            if fmax <= fmin + 1e-8:
                                continue
                            fields_u8 = np.clip(
                                ((fields - fmin) / (fmax - fmin) * 255.0).round(),
                                0,
                                255,
                            ).astype(np.uint8)
                            S_u8 = np.clip(fields_u8.sum(axis=0), 0, 255).astype(np.uint8)
                            fields32 = fields_u8.astype(np.float32)
                            S32 = S_u8.astype(np.float32) + 1e-7
                            w = fields32 / S32[None, ...]
                            wC, wM, wY, wK = w[0], w[1], w[2], w[3]
                            R = (1 - wM) * (1 - wK)
                            G = (1 - wY) * (1 - wK)
                            B = (1 - wC) * (1 - wK)
                            RGBu = np.clip(
                                np.stack([R, G, B], axis=-1), 0, 1
                            ).astype(np.float32)
                            act = float(S_u8.mean()) / 255.0
                            var = float(
                                np.var(RGBu.reshape(-1, 3), axis=0).mean()
                            )
                            sc = 0.6 * act + 0.4 * var
                            if best is None or sc > best:
                                best = sc
                                best_params = (o, a, b)

    dz = (Z0_RANGE[1] - Z0_RANGE[0]) / (Z0_STEPS - 1)
    dw = (W0_RANGE[1] - W0_RANGE[0]) / (W0_STEPS - 1) if W0_STEPS > 1 else 0.0
    ds = (SLOPES[1] - SLOPES[0]) if len(SLOPES) > 1 else 0.0
    return best_params, (dz / 2.0, dw / 2.0, ds / 2.0)


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + eps
    a /= na
    b = b - (a @ b) * a
    nb = np.linalg.norm(b) + eps
    b /= nb
    return a, b


def pick_perp_axis(a: np.ndarray, b: np.ndarray, seed: int = SEED):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=a.shape).astype(np.float32)
    a1, b1 = orthonormalize(a, b)
    v = v - (v @ a1) * a1 - (v @ b1) * b1
    nv = np.linalg.norm(v) + 1e-8
    return v / nv


def rotate_plane(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis_perp: np.ndarray,
    angle_deg: float,
):
    a1, b1 = orthonormalize(a, b)
    n = axis_perp.copy()
    n = n - (n @ a1) * a1 - (n @ b1) * b1
    n /= np.linalg.norm(n) + 1e-8
    theta = np.deg2rad(angle_deg).astype(np.float32)
    a_rot = np.cos(theta) * a1 + np.sin(theta) * n
    a_rot, b_new = orthonormalize(a_rot, b1)
    return o, a_rot, b_new


# ---------------- Pipeline ----------------
def main() -> Dict[str, Dict[str, List[float]]]:
    t0 = time.time()
    (best_o, best_a, best_b), (dz2, dw2, ds2) = coarse_int8_search(res=RES_COARSE)
    t1 = time.time()

    # bounds
    o_low = best_o.copy()
    o_low[2] -= dz2
    o_low[3] -= dw2
    a_low = best_a.copy()
    a_low[2] -= ds2
    a_low[3] -= ds2
    b_low = best_b.copy()
    b_low[2] -= ds2
    b_low[3] -= ds2

    o_high = best_o.copy()
    o_high[2] += dz2
    o_high[3] += dw2
    a_high = best_a.copy()
    a_high[2] += ds2
    a_high[3] += ds2
    b_high = best_b.copy()
    b_high[2] += ds2
    b_high[3] += ds2

    RGB_low, _, S_low = eval_slice_affine(RES_HI, o_low, a_low, b_low)
    sc_low = score_float32(RGB_low, S_low)
    RGB_high, _, S_high = eval_slice_affine(RES_HI, o_high, a_high, b_high)
    sc_high = score_float32(RGB_high, S_high)

    if sc_high >= sc_low:
        o0, a0, b0, RGB0 = o_high, a_high, b_high, RGB_high
    else:
        o0, a0, b0, RGB0 = o_low, a_low, b_low, RGB_low
    t2 = time.time()

    axis_perp = pick_perp_axis(a0, b0, seed=SEED)

    # evenly space angles over [0, 180) for NUM_ROTATED slices
    angles = np.linspace(0, 180, NUM_ROTATED, endpoint=False)

    # Save coarse density map
    RGBc, Fc, Sc = eval_slice_affine(RES_COARSE, best_o, best_a, best_b)
    dens_map = (Sc / (Sc.max() + 1e-7)).astype(np.float32)
    plt.imsave("/mnt/data/coarse_density_map.png", dens_map, cmap="gray")

    # Save origin slice
    base_path = "/mnt/data/slice_origin.png"
    plt.imsave(base_path, RGB0)

    paths: Dict[str, str] = {
        "origin": base_path,
        "coarse_density": "/mnt/data/coarse_density_map.png",
    }

    # Save rotated slices
    for idx, ang in enumerate(angles, start=1):
        o_r, a_r, b_r = rotate_plane(o0, a0, b0, axis_perp, ang)
        RGB_r, _, _ = eval_slice_affine(RES_HI, o_r, a_r, b_r)
        pth = f"/mnt/data/slice_rot_{idx:02d}_{int(ang)}deg.png"
        plt.imsave(pth, RGB_r)
        paths[f"rot_{idx:02d}"] = pth

    t3 = time.time()

    summary: Dict[str, object] = {
        "timings_s": {
            "coarse_search": t1 - t0,
            "refine": t2 - t1,
            "rotations": t3 - t2,
        },
        "best_params_int8": {
            "o": best_o.tolist(),
            "a": best_a.tolist(),
            "b": best_b.tolist(),
            "half_steps": {
                "dz2": float(dz2),
                "dw2": float(dw2),
                "ds2": float(ds2),
            },
        },
        "chosen_origin": {
            "o": o0.tolist(),
            "a": a0.tolist(),
            "b": b0.tolist(),
        },
        "paths": paths,
    }

    # Save summary JSON
    with open("/mnt/data/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

