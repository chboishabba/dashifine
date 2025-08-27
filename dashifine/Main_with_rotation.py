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


def margin_temperature(scores: np.ndarray) -> np.ndarray:
    """Compute a simple margin-dependent temperature.

    The temperature is ``1 + exp(-margin)`` where ``margin`` is the gap between
    the highest and second highest class score for each pixel.
    The returned temperature has shape ``scores[..., 0:1]`` for easy broadcasting.
    """

    sorted_scores = np.sort(scores, axis=-1)
    margin = sorted_scores[..., -1] - sorted_scores[..., -2]
    tau = 1.0 + np.exp(-margin)
    return tau[..., None]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def main(
    output_dir: str | Path,
    res_hi: int = 64,
    res_coarse: int = 16,
    num_rotated: int = 1,
    z0_steps: int = 1,
    w0_steps: int = 1,
    slopes: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Generate placeholder slices and basic rendering data.

    Besides writing placeholder images to ``output_dir`` this function now
    computes per-pixel class weights using a randomly initialized class loading
    matrix.  The returned dictionary therefore includes the generated paths as
    well as ``density`` and ``class_weights`` arrays for further processing.
    """
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

    tau = margin_temperature(F)
    class_weights = softmax(F / tau, axis=-1)

    return {"paths": paths, "density": density, "class_weights": class_weights}


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
