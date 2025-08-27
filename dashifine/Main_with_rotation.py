"""Dashifine slice renderer with simple 4‑D geometry utilities.

This module exposes a small set of functions that are exercised by the unit
tests.  Only a toy procedural field is implemented – enough to verify that the
geometry helpers behave correctly and that successive rotations lead to
distinct images.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np


# ---------------------------------------------------------------------------
# Basic maths helpers


def gelu(x: np.ndarray) -> np.ndarray:
    """Simple odd activation used in tests."""

    return np.tanh(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""

    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def temperature_from_margin(F_i: np.ndarray) -> float:
    """Temperature schedule used for class weighting."""

    sorted_scores = np.sort(F_i)
    margin = sorted_scores[-1] - sorted_scores[-2]
    return 1.0 + np.exp(-margin)


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalise ``a`` and ``b`` using Gram–Schmidt."""

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b


# ---------------------------------------------------------------------------
# 4‑D rotation and sampling utilities


def rotate_plane_4d(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``.

    ``u`` and ``v`` need not be normalised; they simply define the rotation
    plane.  Components of the inputs that lie in this plane are rotated by
    ``angle_deg`` degrees while orthogonal components remain unchanged.
    """

    u, v = orthonormalize(u, v)
    ang = np.deg2rad(angle_deg)

    def _rot(x: np.ndarray) -> np.ndarray:
        xu = np.dot(x, u)
        xv = np.dot(x, v)
        x_perp = x - xu * u - xv * v
        xr = xu * np.cos(ang) - xv * np.sin(ang)
        yr = xu * np.sin(ang) + xv * np.cos(ang)
        return x_perp + xr * u + yr * v

    return _rot(o), _rot(a), _rot(b)


def rotate_plane(
    o: np.ndarray, a: np.ndarray, b: np.ndarray, axis: np.ndarray, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatible wrapper around :func:`rotate_plane_4d`.

    The previous API expected a single rotation ``axis``.  We use ``a`` together
    with ``axis`` to define the rotation plane and delegate to
    :func:`rotate_plane_4d`.
    """

    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


def sample_slice_points(
    H: int, W: int, origin4: np.ndarray, a4: np.ndarray, b4: np.ndarray
) -> np.ndarray:
    """Map a ``H×W`` pixel grid to 4‑D positions.

    The slice is centred on ``origin4`` with basis vectors ``a4`` and ``b4``
    spanning the pixel grid in the range ``[-1, 1]`` along each axis.
    """

    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = (
        origin4[None, :]
        + grid_x.reshape(-1, 1) * a4[None, :]
        + grid_y.reshape(-1, 1) * b4[None, :]
    )
    return pts


# ---------------------------------------------------------------------------
# Field evaluation


@dataclass
class Center:
    mu: np.ndarray
    sigma: np.ndarray
    w: float


def alpha_eff(
    rho_tilde: np.ndarray, a_min: float = 0.6, a_max: float = 2.2, lam: float = 1.0, eta: float = 0.7
) -> np.ndarray:
    t = np.clip(rho_tilde, 0.0, 1.0) ** eta
    return (1 - lam * t) * a_min + lam * t * a_max


def field_and_classes(
    points4: np.ndarray, centers: Iterable[Center], V: np.ndarray, rho_eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the toy field and class scores at 4‑D positions."""

    pts = points4.astype(np.float32)
    centers_list = list(centers)
    N = len(centers_list)
    HW = pts.shape[0]

    g = np.zeros((HW, N), dtype=np.float32)
    for j, c in enumerate(centers_list):
        r = np.linalg.norm((pts - c.mu) / (c.sigma + 1e-8), axis=1)
        g[:, j] = c.w * gelu(1.0 - r)
    g = np.maximum(g, 0.0)

    rho = np.sum(g, axis=1)
    rho_tilde = rho / (np.max(rho) + rho_eps)

    g2 = np.zeros_like(g)
    a_eff = alpha_eff(rho_tilde)
    for j, c in enumerate(centers_list):
        r = np.linalg.norm((pts - c.mu) / (c.sigma + 1e-8), axis=1)
        g2[:, j] = c.w * gelu(a_eff * (1.0 - r))
    g2 = np.maximum(g2, 0.0)

    F = g2 @ V.T
    rho_final = np.sum(g2, axis=1)
    return rho_final, F


# ---------------------------------------------------------------------------
# Colour utilities


def mix_cmy_to_rgb(weights: np.ndarray) -> np.ndarray:
    cmy = np.clip(weights[..., :3], 0.0, 1.0)
    k = np.clip(weights[..., 3:4], 0.0, 1.0)
    rgb = (1.0 - cmy) * (1.0 - k)
    return np.clip(rgb, 0.0, 1.0)


def density_to_alpha(density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    density = np.clip(density, 0.0, 1.0)
    return np.power(density, beta)


def composite_rgb_alpha(
    rgb: np.ndarray, alpha: np.ndarray, bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.float32)
    return rgb * alpha[..., None] + bg_arr * (1.0 - alpha[..., None])


def class_weights_to_rgba(
    class_weights: np.ndarray, density: np.ndarray, beta: float = 1.5
) -> np.ndarray:
    k = np.zeros(class_weights.shape[:2] + (1,), dtype=class_weights.dtype)
    weights = np.concatenate([class_weights[..., :3], k], axis=-1)
    rgb = mix_cmy_to_rgb(weights)
    alpha = density_to_alpha(density, beta)
    return composite_rgb_alpha(rgb, alpha)


def p_adic_address_to_hue_saturation(
    addresses: np.ndarray, depth: np.ndarray, base: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    addresses = addresses.astype(np.int64)
    depth = depth.astype(np.float32)
    if addresses.size == 0:
        return (
            np.empty_like(addresses, dtype=np.float32),
            np.empty_like(depth, dtype=np.float32),
        )

    max_power = int(np.ceil(np.log(addresses.max() + 1) / np.log(base)))
    hue = np.zeros_like(addresses, dtype=np.float32)
    for k in range(max_power):
        digit = (addresses // (base ** k)) % base
        hue += digit / (base ** (k + 1))

    saturation = (
        depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    )
    return hue, saturation


def render(
    addresses: np.ndarray,
    depth: np.ndarray,
    *,
    palette: str = "gray",
    base: int = 2,
) -> np.ndarray:
    if palette == "p_adic":
        hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base=base)
        hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
        return hsv_to_rgb(hsv)

    # default grayscale based on normalised depth
    depth_n = depth / (np.max(depth) + 1e-8) if np.any(depth) else depth
    return np.stack([depth_n] * 3, axis=-1)


# ---------------------------------------------------------------------------
# Rendering pipeline


def _demo_centers() -> Tuple[List[Center], np.ndarray]:
    mus = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.5, 0.0],
            [-0.4, 0.8, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    sigmas = np.full_like(mus, 1.0, dtype=np.float32)
    weights = np.array([1.0, 0.9, 0.8], dtype=np.float32)

    centers: List[Center] = []
    for mu, sigma, w in zip(mus, sigmas, weights):
        centers.append(Center(mu=mu, sigma=sigma, w=float(w)))

    V = np.eye(3, len(centers), dtype=np.float32)  # map centres to CMY
    return centers, V


def render_slice(
    res: int,
    origin4: np.ndarray,
    a4: np.ndarray,
    b4: np.ndarray,
    centers: List[Center],
    V: np.ndarray,
) -> np.ndarray:
    pts = sample_slice_points(res, res, origin4, a4, b4)
    rho, F = field_and_classes(pts, centers, V)
    weights = np.zeros_like(F)
    for i in range(F.shape[0]):
        tau = temperature_from_margin(F[i])
        weights[i] = softmax(F[i] / tau, axis=0)
    rgb = class_weights_to_rgba(
        weights.reshape(res, res, -1), rho.reshape(res, res)
    )
    return rgb


def main(
    output_dir: str | Path,
    res_hi: int = 128,
    res_coarse: int = 32,
    num_rotated: int = 4,
    z0_steps: int = 1,
    w0_steps: int = 1,
    slopes: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Render a small set of slices and return their file paths."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = np.zeros((res_coarse, res_coarse), dtype=np.float32)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    a, b = orthonormalize(a, b)

    # Rotation plane (x-z)
    u = a.copy()
    v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    u, v = orthonormalize(u, v)

    centers, V = _demo_centers()

    paths: Dict[str, str] = {}

    img0 = render_slice(res_hi, o, a, b, centers, V)
    origin_path = out_dir / "slice_origin.png"
    plt.imsave(origin_path, img0)
    paths["origin"] = str(origin_path)

    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        _o, a_r, b_r = rotate_plane_4d(o, a, b, u, v, angle)
        img = render_slice(res_hi, _o, a_r, b_r, centers, V)
        rot_path = out_dir / f"slice_rot_{int(angle):+d}deg.png"
        plt.imsave(rot_path, img)
        paths[f"rot_{angle:+.1f}"] = str(rot_path)

    paths["coarse_density"] = str(density_path)
    return {"paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine slice renderer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=128)
    parser.add_argument("--res_coarse", type=int, default=32)
    parser.add_argument("--num_rotated", type=int, default=4)
    parser.add_argument("--z0_steps", type=int, default=1)
    parser.add_argument("--w0_steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - manual execution only
    args = _parse_args()
    main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        z0_steps=args.z0_steps,
        w0_steps=args.w0_steps,
    )

