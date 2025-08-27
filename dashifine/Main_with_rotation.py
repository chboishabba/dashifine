import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Field definition
# -----------------------------------------------------------------------------
#
# ``mu``   : (N, 2) centre positions in the x/y plane.
# ``sigma``: (N, 2) per-axis standard deviations describing an anisotropic
#            falloff around each centre.
# ``w``    : (N,) weights controlling each centre's contribution.
#
# These constants provide a tiny synthetic field that the demo script samples
# when producing its density maps and rotated slices.

MU = np.array(
    [
        [-0.5, -0.5],
        [0.5, -0.3],
        [0.0, 0.6],
    ],
    dtype=np.float32,
)

SIGMA = np.array(
    [
        [0.3, 0.2],
        [0.25, 0.35],
        [0.2, 0.25],
    ],
    dtype=np.float32,
)

W = np.array([1.0, 0.8, 1.2], dtype=np.float32)

# Exponent for visibility normalisation
BETA = 0.5


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


def _field_density(res: int, beta: float = BETA) -> np.ndarray:
    """Evaluate the synthetic field on a ``res``×``res`` grid.

    Parameters
    ----------
    res:
        Resolution of the square grid to evaluate.
    beta:
        Exponent for visibility normalisation.

    Returns
    -------
    np.ndarray
        Normalised density ``rho_tilde`` raised to ``beta``.
    """

    # Generate grid coordinates in [-1, 1]
    lin = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing="xy")
    pos = np.stack([X, Y], axis=-1)  # (res, res, 2)

    # Compute anisotropic distances r_i for each centre
    diff = pos[None, ...] - MU[:, None, None, :]  # (N, res, res, 2)
    r = np.sqrt(((diff / SIGMA[:, None, None, :]) ** 2).sum(axis=-1))  # (N, res, res)

    # Initial kernel contributions g_i
    g = W[:, None, None] * gelu(1.0 - r)
    rho_tilde = g.sum(axis=0)

    # Mass-coupling via effective alpha
    alpha_eff = 1.0 / (1.0 + rho_tilde)
    g = W[:, None, None] * gelu(alpha_eff - r)
    rho_tilde = g.sum(axis=0)

    # Normalise and compute visibility alpha
    rho_tilde = (rho_tilde - rho_tilde.min()) / (rho_tilde.max() - rho_tilde.min() + 1e-8)
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
) -> Dict[str, Any]:
    """Generate synthetic slices and return their file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = _field_density(res_coarse)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    origin_alpha = _field_density(res_hi)
    origin = np.dstack([origin_alpha] * 3)
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
        img_alpha = _field_density(res_hi)
        img = np.dstack([img_alpha] * 3)
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
