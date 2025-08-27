import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Simple odd activation used for testing."""
    return np.tanh(x)


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalize vectors ``a`` and ``b`` with Gram-Schmidt."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b


def rotate_plane(o: np.ndarray, a: np.ndarray, b: np.ndarray, axis: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate vector ``a`` toward ``axis`` by ``angle_deg`` degrees."""
    angle = np.deg2rad(angle_deg)
    a = a / np.linalg.norm(a)
    axis = axis / np.linalg.norm(axis)
    a_rot = a * np.cos(angle) + axis * np.sin(angle)
    b_new = b / np.linalg.norm(b)
    return o, a_rot, b_new


def p_adic_address_to_hue_saturation(
    addresses: np.ndarray, depth: np.ndarray, base: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Map p-adic addresses to hue and depth to saturation.

    Parameters
    ----------
    addresses:
        Integer array giving the p-adic address for each pixel.
    depth:
        Float array giving the depth for each pixel.
    base:
        Base ``p`` of the p-adic numbers.

    Returns
    -------
    hue, saturation : tuple[np.ndarray, np.ndarray]
        Normalised hue and saturation arrays in ``[0, 1]``.
    """

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
    """Render an RGB image from ``addresses`` and ``depth``.

    ``addresses`` and ``depth`` supply per-pixel p-adic addresses and depth
    values respectively. When ``palette='p_adic'`` the addresses determine the
    hue and the depth controls saturation. Other palettes fall back to a simple
    grayscale mapping of ``depth``.

    Returns
    -------
    np.ndarray
        RGB image with values in ``[0, 1]``.
    """

    if palette == "p_adic":
        hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base)
        hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
        return hsv_to_rgb(hsv)

    value = depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    return np.stack([value, value, value], axis=-1)


def main(
    output_dir: str | Path,
    res_hi: int = 64,
    res_coarse: int = 16,
    num_rotated: int = 1,
    z0_steps: int = 1,
    w0_steps: int = 1,
    slopes: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Generate placeholder slices and return their file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = np.zeros((res_coarse, res_coarse), dtype=np.float32)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    origin = np.zeros((res_hi, res_hi, 3), dtype=np.float32)
    origin_path = out_dir / "slice_origin.png"
    plt.imsave(origin_path, origin)

    paths = {"origin": str(origin_path), "coarse_density": str(density_path)}

    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        _o, _a, _b = rotate_plane(o, a, b, axis, angle)
        img = np.zeros((res_hi, res_hi, 3), dtype=np.float32)
        rot_path = out_dir / f"slice_rot_{int(angle):+d}deg.png"
        plt.imsave(rot_path, img)
        paths[f"rot_{angle:+.1f}"] = str(rot_path)

    return {"paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate placeholder slices")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=64)
    parser.add_argument("--res_coarse", type=int, default=16)
    parser.add_argument("--num_rotated", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
    )
