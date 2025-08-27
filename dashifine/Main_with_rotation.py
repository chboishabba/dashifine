import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
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


def mix_cmy_to_rgb(weights: np.ndarray) -> np.ndarray:
    """Mix the first three channels (CMY) and convert to RGB.

    Parameters
    ----------
    weights:
        Array of shape ``(..., 4)`` containing CMYK weights in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        RGB image in ``[0, 1]``.
    """
    cmy = np.clip(weights[..., :3], 0.0, 1.0)
    k = np.clip(weights[..., 3:4], 0.0, 1.0)
    rgb = (1.0 - cmy) * (1.0 - k)
    return np.clip(rgb, 0.0, 1.0)


def density_to_alpha(density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    """Map density values to opacity using ``density ** beta``."""
    density = np.clip(density, 0.0, 1.0)
    return np.power(density, beta)


def composite_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray, bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Composite an RGB image against a background using the supplied alpha."""
    bg_arr = np.asarray(bg, dtype=np.float32)
    return rgb * alpha[..., None] + bg_arr * (1.0 - alpha[..., None])


def main(
    output_dir: str | Path,
    res_hi: int = 64,
    res_coarse: int = 16,
    num_rotated: int = 1,
    z0_steps: int = 1,
    w0_steps: int = 1,
    slopes: np.ndarray | None = None,
    opacity_exp: float = 1.5,
) -> Dict[str, Any]:
    """Generate example slices and return their file paths."""
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
    rgb = mix_cmy_to_rgb(weights_hi)
    density_hi = weights_hi.mean(axis=-1)
    alpha = density_to_alpha(density_hi, opacity_exp)
    origin = composite_rgb_alpha(rgb, alpha)
    origin_path = out_dir / "slice_origin.png"
    plt.imsave(origin_path, origin)

    paths = {"origin": str(origin_path), "coarse_density": str(density_path)}

    # Generate rotated slices (placeholder using 90-degree rotations)
    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        _o, _a, _b = rotate_plane(o, a, b, axis, angle)
        rgb_rot = np.rot90(rgb, k=i % 4, axes=(0, 1))
        alpha_rot = np.rot90(alpha, k=i % 4, axes=(0, 1))
        img = composite_rgb_alpha(rgb_rot, alpha_rot)
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
    parser.add_argument("--opacity_exp", type=float, default=1.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        opacity_exp=args.opacity_exp,
    )
