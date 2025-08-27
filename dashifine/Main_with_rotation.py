"""Core rendering utilities for the Dashifine demos.

This module provides a minimal set of primitives used throughout the tests. It
offers simple geometric helpers, colour mapping utilities and a tiny demo
``main`` function which mirrors the behaviour of the stand‑alone patch module.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import hashlib
import re
from matplotlib.colors import hsv_to_rgb

# ------------------------------ basic primitives -----------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """Simple odd activation used in tests."""
    return np.tanh(x)


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalise vectors ``a`` and ``b`` with Gram–Schmidt."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b


def rotate_plane(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``(o, a, b)`` around ``axis`` using :func:`rotate_plane_4d`."""

    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


def rotate_plane_4d(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``.

    The plane is defined by two (not necessarily normalised) vectors ``u`` and
    ``v``.  Any component of the inputs lying in this plane is rotated by
    ``angle_deg`` degrees while the orthogonal component is left unchanged.
    """
def rotate_plane_4d(o: np.ndarray, a: np.ndarray, b: np.ndarray, u: np.ndarray, v: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``."""
    u, v = orthonormalize(u, v)
    angle = np.deg2rad(angle_deg)

    def _rotate(x: np.ndarray) -> np.ndarray:
        xu = np.dot(x, u)
        xv = np.dot(x, v)
        x_perp = x - xu * u - xv * v
        xr = xu * np.cos(angle) - xv * np.sin(angle)
        yr = xu * np.sin(angle) + xv * np.cos(angle)
        return x_perp + xr * u + yr * v

    return _rotate(o), _rotate(a), _rotate(b)



def rotate_plane(o: np.ndarray, a: np.ndarray, b: np.ndarray, axis: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatible wrapper for :func:`rotate_plane_4d`."""
    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


# ----------------------------- colour utilities ------------------------------

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
    """Map a p-adic style address string to HSV components.

    The integer portion of ``addr_digits`` is interpreted as base-``p`` digits
    contributing fractional hue.  Any leading digits form a *prefix* which is
    hashed to provide a stable base hue.  An optional fractional part encodes
    depth, modulating saturation and value.

    Parameters
    ----------
    addr_digits:
        Address string of the form ``"<prefix><digits>[.<depth>]"``.
    base:
        Base ``p`` used to interpret the integer suffix.

    Returns
    -------
    tuple[float, float, float]
        Normalised ``(h, s, v)`` components.
    """

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
    """Placeholder eigen palette mapping to grayscale.

    Parameters
    ----------
    weights:
        Array of shape ``(..., C)`` containing class weights."""

def class_weights_to_rgba(
    class_weights: np.ndarray,
    density: np.ndarray,
    beta: float = 1.5,
) -> np.ndarray:
    """Map class weights and density to a composited RGB image.

    The first three channels of ``class_weights`` are interpreted as CMY
    contributions.  A zero ``K`` channel is appended and the result converted to
    RGB.  Opacity is computed as ``density ** beta`` and the RGB image is
    composited over a white background.

    Parameters
    ----------
    class_weights:
        Array of shape ``(H, W, C)`` with ``C >= 3`` containing per-class
        weights.
    density:
        Array of shape ``(H, W)`` giving normalised density ``rho_tilde``.
    beta:
        Exponent controlling opacity from density.

    Returns
    -------
    np.ndarray
        Composited RGB image in ``[0, 1]``.
    """


def class_weights_to_rgba(class_weights: np.ndarray, density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    """Map class weights and density to a composited RGB image."""
    k = np.zeros(class_weights.shape[:2] + (1,), dtype=class_weights.dtype)
    weights = np.concatenate([class_weights[..., :3], k], axis=-1)
    rgb = mix_cmy_to_rgb(weights)
    alpha = density_to_alpha(density, beta)
    return composite_rgb_alpha(rgb, alpha)


def p_adic_address_to_hue_saturation(
    addresses: np.ndarray, depth: np.ndarray, base: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Map p-adic addresses to hue and depth to saturation."""



# ---------------------------- p-adic visualisation ---------------------------

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

    saturation = (
        depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    )
    return hue, saturation


def render(addresses: np.ndarray, depth: np.ndarray, *, palette: str = "gray", base: int = 2) -> np.ndarray:
    """Render an RGB image from ``addresses`` and ``depth``."""
    if palette == "p_adic":
        hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base)
        hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
        return hsv_to_rgb(hsv)

    value = depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    return np.stack([value, value, value], axis=-1)


def _field_density(
    res: int,
    *,
    centers: FieldCenters = CENTERS,
    beta: float = BETA,
) -> np.ndarray:
    """Evaluate the synthetic field on a ``res``×``res`` grid.

    Parameters
    ----------
    res:
        Resolution of the square grid to evaluate.
    centers:
        ``FieldCenters`` describing positions, falloff and weights of kernels.
    beta:
        Exponent for visibility normalisation.

    Returns
    -------
    np.ndarray
        Visibility ``alpha_vis`` derived from the normalised density.
    """

    mu, sigma, w = centers.mu, centers.sigma, centers.w

    # Generate grid coordinates in [-1, 1]
    lin = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing="xy")
    pos = np.stack([X, Y], axis=-1)  # (res, res, 2)

    # Compute anisotropic distances r_i for each centre
    diff = pos[None, ...] - mu[:, None, None, :]  # (N, res, res, 2)
    ri = np.linalg.norm(diff / sigma[:, None, None, :], axis=-1)  # (N, res, res)

    # Initial kernel contributions and normalised density
    g = w[:, None, None] * gelu(1.0 - ri)
    rho = g.sum(axis=0)
    rho_tilde = (rho - rho.min()) / (rho.max() - rho.min() + 1e-8)

    # Mass-coupling via effective alpha
    alpha_eff = 1.0 / (1.0 + rho_tilde)
    g = w[:, None, None] * gelu(alpha_eff * (1.0 - ri))
    rho = g.sum(axis=0)

    # Normalise and compute visibility alpha
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
    opacity_exp: float = 1.5,
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

    origin_alpha = _field_density(res_hi, centers=centers, beta=beta)
    origin = np.dstack([origin_alpha] * 3)
    
    """Generate example slices and return their file paths"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Generate coarse density for reference
    x = np.linspace(0.0, 1.0, res_coarse, dtype=np.float32)
    y = np.linspace(0.0, 1.0, res_coarse, dtype=np.float32)
    Xc, Yc = np.meshgrid(x, y)
    weights_coarse = np.stack([Xc, Yc, 1.0 - Xc, 0.5 * np.ones_like(Xc)], axis=-1)
    density = weights_coarse.mean(axis=-1)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    # High-resolution origin slice
    xh = np.linspace(0.0, 1.0, res_hi, dtype=np.float32)
    yh = np.linspace(0.0, 1.0, res_hi, dtype=np.float32)
    Xh, Yh = np.meshgrid(xh, yh)
    weights_hi = np.stack([Xh, Yh, 1.0 - Xh, 0.5 * np.ones_like(Xh)], axis=-1)
    if palette == "cmy":
        rgb = mix_cmy_to_rgb(weights_hi)
    elif palette == "eigen":
        rgb = eigen_palette(weights_hi)
    elif palette == "lineage":
        num_classes = weights_hi.shape[-1]
        palette_rgb = np.zeros((num_classes, 3), dtype=np.float32)
        for i in range(num_classes):
            h, s, v = lineage_hsv_from_address(str(i))
            palette_rgb[i] = hsv_to_rgb([h, s, v])
        weights_norm = weights_hi / (
            np.sum(weights_hi, axis=-1, keepdims=True) + 1e-8
        )
        rgb = weights_norm @ palette_rgb
    else:
        rgb = mix_cmy_to_rgb(weights_hi)
    density_hi = weights_hi.mean(axis=-1)
    alpha = density_to_alpha(density_hi, opacity_exp)
    origin = composite_rgb_alpha(rgb, alpha)
    
    """Generate placeholder slices and basic rendering data.

    Besides writing placeholder images to ``output_dir`` this function now
    computes per-pixel class weights using a randomly initialized class loading
    matrix.  The returned dictionary therefore includes the generated paths as
    well as ``density`` and ``class_weights`` arrays for further processing.    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density_path = out_dir / "coarse_density_map.png"
    origin_path = out_dir / "slice_origin.png"

    paths = {"origin": str(origin_path), "coarse_density": str(density_path)}

    # Generate rotated slices using 4D plane rotations
    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # Generate origin and coarse density maps using the field evaluation
    coarse_points = sample_slice_image(o, a, b, res_coarse)
    density = np.mean(eval_field(coarse_points), axis=-1)
    plt.imsave(density_path, density, cmap="gray")

    origin_points = sample_slice_image(o, a, b, res_hi)
    origin_img = eval_field(origin_points)
    plt.imsave(origin_path, origin_img)

    # Define rotation plane and generate rotated slices
    rot_u = a + b
    rot_v = axis

    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        _o, _a, _b = rotate_plane_4d(o, a, b, rot_u, rot_v, angle)
        img_alpha = _field_density(res_hi, centers=centers, beta=beta)
        img = np.dstack([img_alpha] * 3)
        rgb_rot = np.rot90(rgb, k=i % 4, axes=(0, 1))
        alpha_rot = np.rot90(alpha, k=i % 4, axes=(0, 1))
        img = composite_rgb_alpha(rgb_rot, alpha_rot)

        _o, _a, _b = rotate_plane_4d(o, a, b, a, axis, angle)
        points = sample_slice_image(_o, _a, _b, res_hi)
        img = eval_field(points)
        rot_path = out_dir / f"slice_rot_{int(angle):+d}deg.png"
        plt.imsave(rot_path, img)
        paths[f"rot_{angle:+.1f}"] = str(rot_path)

    # ------------------------------------------------------------------
    # Simple class weight computation for each coarse pixel
    # ------------------------------------------------------------------
    xs, ys = np.meshgrid(
        np.linspace(-1.0, 1.0, res_coarse, dtype=np.float32),
        np.linspace(-1.0, 1.0, res_coarse, dtype=np.float32),
        indexing="ij",
    )
    g = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    # Random class loading matrix ``V``
    num_classes = 3
    V = np.random.randn(num_classes, g.shape[-1]).astype(np.float32)
    F = g @ V.T
    F = F.reshape(res_coarse, res_coarse, num_classes)

    # Per-pixel temperature from score margins followed by softmax.
    tau = np.apply_along_axis(temperature_from_margin, -1, F)[..., None]
    class_weights = softmax(F / tau, axis=-1)
    class_img = class_weights_to_rgba(class_weights, density, opacity_exp)
    class_path = out_dir / "class_weights_composite.png"
    plt.imsave(class_path, class_img)
    paths["class_weights"] = str(class_path)
# ------------------------------ placeholder main ----------------------------

def main(output_dir: str | Path, **_: Any) -> Dict[str, Any]:
    """Minimal entry point used in tests."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {"paths": {}}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine placeholder script")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=64)
    parser.add_argument("--res_coarse", type=int, default=16)
    parser.add_argument("--num_rotated", type=int, default=1)
    parser.add_argument("--opacity_exp", type=float, default=1.5)
    parser.add_argument(
        "--palette",
        type=str,
        default="cmy",
        choices=["cmy", "lineage", "eigen"],
        help="Colour palette for slice rendering ('cmy', 'lineage', or 'eigen')",
    )

    return parser.parse_args()


if __name__ == "main":  # pragma: no cover - defensive
    args = _parse_args()
    main(output_dir=args.output_dir)
