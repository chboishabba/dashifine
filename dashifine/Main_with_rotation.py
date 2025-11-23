from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

# NOTE: The small helpers in this file are implementation hooks for the formal
# specifications in ``formal/``.  In particular, the p-adic colouring and
# branch-selection placeholders are meant to track the guarantees proved in
# ``FormalCore.PAdic`` and ``FormalCore.Tetralemma``.


@dataclass
class FieldCenters:
    """Placeholder container for field centre parameters."""

    mu: np.ndarray
    sigma: np.ndarray
    w: np.ndarray


# Default centres and opacity exponent used by tests and examples
CENTERS = FieldCenters(
    mu=np.zeros((1, 2), dtype=np.float32),
    sigma=np.ones((1, 2), dtype=np.float32),
    w=np.ones(1, dtype=np.float32),
)
BETA: float = 1.5


def gelu(x: np.ndarray) -> np.ndarray:
    """Tiny odd activation used in a couple of tests."""

    return np.tanh(x)


def orthonormalize(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal basis spanning ``a`` and ``b``."""

    ao = a / np.linalg.norm(a)
    b_proj = b - np.dot(b, ao) * ao
    bo = b_proj / np.linalg.norm(b_proj)
    return ao, bo


def rotate_plane_4d(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate the plane ``(a, b)`` around the plane ``(u, v)`` by ``angle`` degrees."""

    uo, vo = orthonormalize(u, v)
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)

    def rot_vec(x: np.ndarray) -> np.ndarray:
        alpha = np.dot(x, uo)
        beta = np.dot(x, vo)
        plane = (alpha * c - beta * s) * uo + (alpha * s + beta * c) * vo
        ortho = x - alpha * uo - beta * vo
        return ortho + plane

    return rot_vec(o), rot_vec(a), rot_vec(b)


def rotate_plane(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper rotating ``(a, b)`` around ``axis``."""

    return rotate_plane_4d(o, a, b, a, axis, angle)


def mix_cmy_to_rgb(weights: np.ndarray) -> np.ndarray:
    cmy = np.clip(weights[..., :3], 0.0, 1.0)
    if weights.shape[-1] > 3:
        k = np.clip(weights[..., 3:4], 0.0, 1.0)
    else:
        k = 0.0
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
    bg_arr = np.asarray(bg, dtype=np.float32)
    return rgb * alpha[..., None] + bg_arr * (1.0 - alpha[..., None])


def class_weights_to_rgba(
    class_weights: np.ndarray,
    density: np.ndarray,
    beta: float = BETA,
) -> np.ndarray:
    """Map class weights and density to a composited RGB image."""

    cmy = np.zeros((*class_weights.shape[:2], 3), dtype=np.float32)
    cmy[..., : class_weights.shape[-1]] = class_weights[..., :3]
    rgb = mix_cmy_to_rgb(cmy)
    alpha = density_to_alpha(density, beta)
    return composite_rgb_alpha(rgb, alpha)


def p_adic_address_to_hue_saturation(
    addresses: np.ndarray, depth: np.ndarray, base: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Map p-adic ``addresses`` and ``depth`` to hue and saturation."""

    # Spec note: this indexing structure is intended to satisfy the digitwise
    # stability properties discussed in ``FormalCore.PAdic.geom_sum_3adic``—we
    # treat the addresses as convergent p-adic expansions so the hue accumulates
    # monotonically as more digits are introduced.

    addresses = addresses.astype(np.int64)
    depth = depth.astype(np.float32)
    if addresses.size == 0:
        return (
            np.empty_like(addresses, dtype=np.float32),
            np.empty_like(depth, dtype=np.float32),
        )

    max_power = (
        int(np.ceil(np.log(addresses.max() + 1) / np.log(base))) if np.any(addresses) else 1
    )
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
    """Render an RGB image given ``addresses`` and ``depth``."""

    if palette == "p_adic":
        hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base=base)
        hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
        return hsv_to_rgb(hsv)

    norm = depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    return np.stack([norm, norm, norm], axis=-1)


def main(
    output_dir: str | Path,
    res_hi: int = 4,
    res_coarse: int = 2,
    num_rotated: int = 1,
    z0_steps: int = 1,
    w0_steps: int = 1,
    slopes: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Generate a couple of tiny placeholder images."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((res_hi, res_hi, 3), dtype=np.float32)
    slice_path = out_dir / "slice.png"
    plt.imsave(slice_path, img)

    coarse = np.zeros((res_coarse, res_coarse), dtype=np.float32)
    # Spec note: the uniform coarse grid mirrors the balanced prior underpinning
    # ``FormalCore.Tetralemma.branch_threshold``—each cell contributes equally so
    # a 0.5 marginal corresponds to a decisive branch in the formal model.
    coarse_path = out_dir / "coarse.png"
    plt.imsave(coarse_path, coarse, cmap="gray")

    return {"paths": {"slice": str(slice_path), "coarse": str(coarse_path)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="examples")
    args = parser.parse_args()
    main(args.output_dir)
