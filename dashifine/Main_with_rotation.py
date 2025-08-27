"""Dashifine demo utilities.

This module contains a tiny synthetic 4D field along with a number of helper
functions used throughout the tests.  The implementation is intentionally small
and is not meant to be a feature complete renderer – many of the operations are
simple placeholders that nevertheless exercise the control flow of the real
project.

The script can also be executed directly to generate a couple of example images
showcasing the different colour palettes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np


# ---------------------------------------------------------------------------
# Basic data structures
# ---------------------------------------------------------------------------


@dataclass
class FieldCenters:
    """Parameterisation of a small synthetic field."""

    mu: np.ndarray
    """Centre positions with shape ``(N, 4)``."""

    sigma: np.ndarray
    """Per-axis standard deviations for anisotropic falloff, shape ``(N, 4)``."""

    w: np.ndarray
    """Weights controlling each centre's contribution with shape ``(N,)``."""


# A few hard coded centres used by tests
CENTERS = FieldCenters(
    mu=np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [-0.5, 0.25, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
    sigma=np.array(
        [
            [0.3, 0.3, 0.3, 0.3],
            [0.25, 0.25, 0.25, 0.25],
            [0.35, 0.35, 0.35, 0.35],
        ],
        dtype=np.float32,
    ),
    w=np.array([1.0, 0.8, 1.2], dtype=np.float32),
)

# Exponent used when converting density to opacity
BETA = 1.5


# ---------------------------------------------------------------------------
# Small maths helpers
# ---------------------------------------------------------------------------


def gelu(x: np.ndarray) -> np.ndarray:
    """Light‑weight GELU approximation used in a few tests."""

    return np.tanh(x)


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalise vectors ``a`` and ``b`` using Gram–Schmidt."""

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b


def rotate_plane_4d(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``."""

    u, v = orthonormalize(u, v)
    theta = np.deg2rad(angle_deg)

    def _rot(x: np.ndarray) -> np.ndarray:
        xu = np.dot(x, u)
        xv = np.dot(x, v)
        x_perp = x - xu * u - xv * v
        xr = xu * np.cos(theta) - xv * np.sin(theta)
        yr = xu * np.sin(theta) + xv * np.cos(theta)
        return x_perp + xr * u + yr * v

    return _rot(o), _rot(a), _rot(b)


def rotate_plane(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward‑compatible wrapper using ``a`` and ``axis`` as rotation plane."""

    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


# ---------------------------------------------------------------------------
# Slice sampling and simple field evaluation
# ---------------------------------------------------------------------------


def sample_slice_image(o: np.ndarray, a: np.ndarray, b: np.ndarray, res: int) -> np.ndarray:
    """Map pixel coordinates of a slice image to 4D positions."""

    xs = np.linspace(-0.5, 0.5, res, endpoint=False, dtype=np.float32) + 0.5 / res
    ys = np.linspace(-0.5, 0.5, res, endpoint=False, dtype=np.float32) + 0.5 / res
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = o + grid_x[..., None] * a + grid_y[..., None] * b
    return points.astype(np.float32)


def eval_field(points: np.ndarray) -> np.ndarray:
    """Evaluate a simple CMYK‑style field at 4D ``points``."""

    centers = np.eye(4, dtype=np.float32)
    dists = np.linalg.norm(points[..., None, :] - centers[None, None, :, :], axis=-1)
    cmyk = gelu(1.0 - dists)
    rgb = 1.0 - cmyk[..., :3]
    return np.clip(rgb, 0.0, 1.0)


def temperature_from_margin(F_i: np.ndarray) -> float:
    """Compute a softmax temperature from the score margin of a pixel."""

    sorted_scores = np.sort(F_i)
    margin = sorted_scores[-1] - sorted_scores[-2]
    return 1.0 + np.exp(-margin)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def mix_cmy_to_rgb(weights: np.ndarray) -> np.ndarray:
    """Mix CMY(K) weights to RGB."""

    cmy = np.clip(weights[..., :3], 0.0, 1.0)
    k = np.clip(weights[..., 3:4], 0.0, 1.0)
    rgb = (1.0 - cmy) * (1.0 - k)
    return np.clip(rgb, 0.0, 1.0)


def density_to_alpha(density: np.ndarray, beta: float = BETA) -> np.ndarray:
    density = np.clip(density, 0.0, 1.0)
    return np.power(density, beta)


def composite_rgb_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Composite an RGB image against ``bg`` using the supplied alpha."""

    bg_arr = np.asarray(bg, dtype=np.float32)
    return rgb * alpha[..., None] + bg_arr * (1.0 - alpha[..., None])


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------


def lineage_hue_from_address(addr: str) -> float:
    """Return a deterministic hue in ``[0, 1]`` for ``addr``.

    The current implementation hashes the address string and uses the hash to
    generate a stable hue.  The exact mapping is unimportant for the tests – it
    merely needs to be deterministic.
    """

    h = hash(addr) & 0xFFFFFFFF
    return (h / 0xFFFFFFFF) % 1.0


def eigen_palette(W: np.ndarray) -> np.ndarray:
    """Project class weights to their first three principal components."""

    if W.ndim == 3:
        flat = W.reshape(-1, W.shape[-1])
    else:
        flat = W
    if flat.size == 0:
        return np.zeros((flat.shape[0], 3), dtype=np.float32)

    Wc = flat - np.mean(flat, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    proj = Wc @ Vt[:3].T
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])), mode="constant")
    mn = proj.min(axis=0, keepdims=True)
    mx = proj.max(axis=0, keepdims=True)
    denom = np.where(mx - mn > 1e-8, mx - mn, 1.0)
    rgb = (proj - mn) / denom
    return np.clip(rgb, 0.0, 1.0)


def class_weights_to_rgba(
    class_weights: np.ndarray,
    density: np.ndarray,
    beta: float = BETA,
) -> np.ndarray:
    """Convert class weights to an RGB image composited on white."""

    k = np.zeros(class_weights.shape[:2] + (1,), dtype=class_weights.dtype)
    weights = np.concatenate([class_weights[..., :3], k], axis=-1)
    rgb = mix_cmy_to_rgb(weights)
    alpha = density_to_alpha(density, beta)
    return composite_rgb_alpha(rgb, alpha)


# ---------------------------------------------------------------------------
# P‑adic helper used by a couple of tests
# ---------------------------------------------------------------------------


def p_adic_address_to_hue_saturation(
    addresses: np.ndarray,
    depth: np.ndarray,
    base: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map p‑adic addresses to hue and depth to saturation."""

    addresses = addresses.astype(np.int64)
    depth = depth.astype(np.float32)
    if addresses.size == 0:
        return (
            np.empty_like(addresses, dtype=np.float32),
            np.empty_like(depth, dtype=np.float32),
        )

    max_power = int(np.ceil(np.log(addresses.max() + 1) / np.log(base))) if np.any(addresses) else 1
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
    """Render an RGB image from ``addresses`` and ``depth``."""

    if palette == "p_adic":
        hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base)
        hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
        return hsv_to_rgb(hsv)

    value = depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    return np.stack([value, value, value], axis=-1)


# ---------------------------------------------------------------------------
# Slice rendering with palette selection
# ---------------------------------------------------------------------------


def render_slice(
    H: int,
    W: int,
    origin4: np.ndarray,
    a4: np.ndarray,
    b4: np.ndarray,
    centers: Any,
    V: np.ndarray,
    palette: str = "cmy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a coloured slice using one of several palettes.

    The implementation is intentionally simple: the returned colours do not
    attempt to represent the underlying field faithfully but they allow the
    tests to exercise the palette selection logic.
    """

    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    weights = np.stack([X, Y, 1.0 - X, 0.5 * np.ones_like(X)], axis=-1)

    if palette.lower() == "eigen":
        rgb = eigen_palette(weights).reshape(H, W, 3)
    elif palette.lower() == "lineage":
        top = np.argmax(weights, axis=-1)
        hsv = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                hue = lineage_hue_from_address(str(int(top[i, j])))
                hsv[i, j] = [hue, 1.0, 1.0]
        rgb = hsv_to_rgb(hsv)
    else:  # default CMY
        rgb = mix_cmy_to_rgb(weights)

    density = weights.mean(axis=-1)
    alpha = density_to_alpha(density, BETA)
    return rgb, alpha


# ---------------------------------------------------------------------------
# Minimal field density and main demo entry point
# ---------------------------------------------------------------------------


def _field_density(res: int, centers: FieldCenters = CENTERS, beta: float = BETA) -> np.ndarray:
    """Evaluate the synthetic field on a ``res``×``res`` grid."""

    mu, sigma, w = centers.mu, centers.sigma, centers.w

    lin = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing="xy")
    pos = np.stack([X, Y, np.zeros_like(X), np.zeros_like(X)], axis=-1)

    diff = pos[None, ...] - mu[:, None, None, :]
    ri = np.linalg.norm(diff / sigma[:, None, None, :], axis=-1)

    g = w[:, None, None] * gelu(1.0 - ri)
    rho = g.sum(axis=0)
    rho_tilde = (rho - rho.min()) / (rho.max() - rho.min() + 1e-8)
    alpha_vis = rho_tilde ** beta
    return alpha_vis


def main(
    output_dir: str | Path,
    res_hi: int = 64,
    res_coarse: int = 16,
    num_rotated: int = 1,
    z0_steps: int = 1,
    w0_steps: int = 1,
    slopes: np.ndarray | None = None,
    opacity_exp: float = BETA,
    palette: str = "cmy",
    centers: FieldCenters = CENTERS,
    beta: float = BETA,
) -> Dict[str, Any]:
    """Generate synthetic slices and return their file paths."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = _field_density(res_coarse, centers=centers, beta=beta)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    # High‑resolution origin slice using the requested palette
    rgb, alpha = render_slice(res_hi, res_hi, np.zeros(4, dtype=np.float32), np.eye(4)[0], np.eye(4)[1], centers, np.eye(3), palette)
    origin = composite_rgb_alpha(rgb, alpha)
    origin_path = out_dir / "slice_origin.png"
    plt.imsave(origin_path, origin)

    paths = {"origin": str(origin_path), "coarse_density": str(density_path)}
    return {"paths": paths}


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate placeholder slices")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=64)
    parser.add_argument("--res_coarse", type=int, default=16)
    parser.add_argument("--num_rotated", type=int, default=1)
    parser.add_argument("--opacity_exp", type=float, default=BETA)
    parser.add_argument(
        "--palette",
        type=str,
        default="cmy",
        choices=["cmy", "lineage", "eigen"],
        help="Colour palette for slice rendering",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        opacity_exp=args.opacity_exp,
        palette=args.palette,
    )

