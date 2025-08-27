"""Core rendering utilities for the Dashifine demos.

This module offers a grab‑bag of small helpers used by the tests.  It contains
basic maths primitives, simple colour utilities and a tiny demo ``main``
function capable of rendering a few rotated 4‑D slices.  The implementation is
deliberately compact; it is not intended to be a full featured renderer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb


# ---------------------------------------------------------------------------
# basic primitives
# ---------------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """Tiny odd activation used in the tests."""

    return np.tanh(x)


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalise vectors ``a`` and ``b`` with Gram–Schmidt."""

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b


# ---------------------------------------------------------------------------
# rotation helpers
# ---------------------------------------------------------------------------

def rotate_plane_4d(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``.

    Any component of the inputs lying in this plane is rotated by
    ``angle_deg`` degrees while the orthogonal component is left unchanged.
    """

    u, v = orthonormalize(u, v)
    angle = np.deg2rad(angle_deg)

    def _rotate(x: np.ndarray) -> np.ndarray:
        xu = float(np.dot(x, u))
        xv = float(np.dot(x, v))
        x_perp = x - xu * u - xv * v
        xr = xu * np.cos(angle) - xv * np.sin(angle)
        yr = xu * np.sin(angle) + xv * np.cos(angle)
        return x_perp + xr * u + yr * v

    return _rotate(o), _rotate(a), _rotate(b)


def rotate_plane(o: np.ndarray, a: np.ndarray, b: np.ndarray, axis: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatible wrapper for :func:`rotate_plane_4d`."""

    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


def sample_slice_image(H: int, W: int, origin4: np.ndarray, a4: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """Map each pixel of an ``H``×``W`` grid to 4‑D coordinates.

    The slice is defined by ``origin4`` and basis vectors ``a4`` and ``b4``. For
    pixel coordinates ``(u, v)`` in ``[-1, 1]`` the mapped point is

    ``x = origin4 + u * a4 + v * b4``.
    """

    u = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="xy")
    pts = origin4[None, None, :] + U[..., None] * a4[None, None, :] + V[..., None] * b4[None, None, :]
    return pts


# ---------------------------------------------------------------------------
# colour utilities
# ---------------------------------------------------------------------------

def mix_cmy_to_rgb(weights: np.ndarray) -> np.ndarray:
    cmy = np.clip(weights[..., :3], 0.0, 1.0)
    k = np.clip(weights[..., 3:4], 0.0, 1.0)
    rgb = (1.0 - cmy) * (1.0 - k)
    return np.clip(rgb, 0.0, 1.0)


def density_to_alpha(density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    density = np.clip(density, 0.0, 1.0)
    return np.power(density, beta)


def composite_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray, bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.float32)
    return rgb * alpha[..., None] + bg_arr * (1.0 - alpha[..., None])


def lineage_hsv_from_address(addr_digits: str, base: int = 4) -> Tuple[float, float, float]:
    """Map a p-adic style address string to HSV components."""

    import hashlib
    import re

    if "." in addr_digits:
        addr_main, frac_part = addr_digits.split(".", 1)
    else:
        addr_main, frac_part = addr_digits, ""

    m = re.match(r"(\d*?)(\d+)$", addr_main)
    if m:
        prefix_digits, suffix_digits = m.group(1), m.group(2)
    else:
        prefix_digits, suffix_digits = "", addr_main

    if prefix_digits:
        h = hashlib.sha256(prefix_digits.encode("utf-8")).hexdigest()
        prefix_hue = int(h[:8], 16) / 0xFFFFFFFF
    else:
        prefix_hue = 0.0

    hue = prefix_hue
    for k, ch in enumerate(reversed(suffix_digits)):
        digit = min(int(ch), base - 1)
        hue += digit / (base ** (k + 1))
    hue %= 1.0

    depth = float(f"0.{frac_part}") if frac_part else 0.0
    saturation = np.clip(depth, 0.0, 1.0)
    value = 1.0 - 0.5 * depth
    return float(hue), float(saturation), float(value)


def eigen_palette(weights: np.ndarray) -> np.ndarray:
    """Placeholder eigen palette mapping to grayscale."""

    g = np.mean(weights, axis=-1, keepdims=True)
    return np.repeat(g, 3, axis=-1)


def class_weights_to_rgba(class_weights: np.ndarray, density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    """Map class weights and density to a composited RGB image."""

    k = np.zeros(class_weights.shape[:2] + (1,), dtype=class_weights.dtype)
    weights = np.concatenate([class_weights[..., :3], k], axis=-1)
    rgb = mix_cmy_to_rgb(weights)
    alpha = density_to_alpha(density, beta)
    return composite_rgb_alpha(rgb, alpha)


# ---------------------------------------------------------------------------
# p-adic visualisation utilities
# ---------------------------------------------------------------------------

def p_adic_address_to_hue_saturation(addresses: np.ndarray, depth: np.ndarray, base: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Map p-adic addresses to hue and depth to saturation."""

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

    saturation = depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    return hue, saturation


def render(addresses: np.ndarray, depth: np.ndarray, *, palette: str = "gray", base: int = 2) -> np.ndarray:
    """Render an RGB image from ``addresses`` and ``depth``."""

    if palette == "p_adic":
        hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base)
        hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
        return hsv_to_rgb(hsv)

    value = depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    return np.stack([value, value, value], axis=-1)


# ---------------------------------------------------------------------------
# tiny demo renderer
# ---------------------------------------------------------------------------

def eval_field(points4: np.ndarray) -> np.ndarray:
    """Simple 4-D field used for demo rendering.

    The field varies along the ``z`` and ``w`` axes which makes rotations in
    those dimensions visually apparent.
    """

    z = points4[..., 2]
    w = points4[..., 3]
    val = np.sin(3.0 * z) + np.cos(3.0 * w)
    val = (val - val.min()) / (val.max() - val.min() + 1e-8)
    return np.stack([val, val, val], axis=-1)


def main(
    output_dir: str | Path,
    res_hi: int = 64,
    num_rotated: int = 1,
    **_: Any,
) -> Dict[str, Any]:
    """Render an origin slice and a number of rotated slices."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # base slice basis vectors
    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # Rotate the slice basis around a plane that mixes the x/y slice with the
    # ``w`` axis so that the field varies as the angle changes.
    u = a + b
    v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # origin slice
    origin_pts = sample_slice_image(res_hi, res_hi, o, a, b)
    origin_img = eval_field(origin_pts)
    origin_path = out / "slice_origin.png"
    plt.imsave(origin_path, origin_img)
    paths: Dict[str, str] = {"origin": str(origin_path)}

    # rotated slices
    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        _o, _a, _b = rotate_plane_4d(o, a, b, u, v, angle)
        pts = sample_slice_image(res_hi, res_hi, _o, _a, _b)
        img = eval_field(pts)
        rot_path = out / f"slice_rot_{int(angle):+d}deg.png"
        plt.imsave(rot_path, img)
        paths[f"rot_{angle:+.1f}"] = str(rot_path)

    return {"paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine demo renderer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=64)
    parser.add_argument("--num_rotated", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - manual testing helper
    args = _parse_args()
    main(output_dir=args.output_dir, res_hi=args.res_hi, num_rotated=args.num_rotated)

