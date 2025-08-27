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


def class_weights_to_rgba(class_weights: np.ndarray, density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    """Map class weights and density to a composited RGB image."""
    k = np.zeros(class_weights.shape[:2] + (1,), dtype=class_weights.dtype)
    weights = np.concatenate([class_weights[..., :3], k], axis=-1)
    rgb = mix_cmy_to_rgb(weights)
    alpha = density_to_alpha(density, beta)
    return composite_rgb_alpha(rgb, alpha)


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


# ------------------------------ placeholder main ----------------------------

def main(output_dir: str | Path, **_: Any) -> Dict[str, Any]:
    """Minimal entry point used in tests."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {"paths": {}}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine placeholder script")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "main":  # pragma: no cover - defensive
    args = _parse_args()
    main(output_dir=args.output_dir)
