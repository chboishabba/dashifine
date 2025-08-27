import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
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


def rotate_plane(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatible wrapper for :func:`rotate_plane_4d`.

    The rotation plane is defined by ``a`` and ``axis``.  This helper exists
    only so older code and tests expecting ``rotate_plane`` continue to work.
    """

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
    u, v = orthonormalize(u, v)
    angle = np.deg2rad(angle_deg)

    def _rotate(x: np.ndarray) -> np.ndarray:
        xu = np.dot(x, u)
        xv = np.dot(x, v)
        # Component orthogonal to the rotation plane remains unchanged
        x_perp = x - xu * u - xv * v
        xr = xu * np.cos(angle) - xv * np.sin(angle)
        yr = xu * np.sin(angle) + xv * np.cos(angle)
        return x_perp + xr * u + yr * v

    return _rotate(o), _rotate(a), _rotate(b)


def sample_slice_image(o: np.ndarray, a: np.ndarray, b: np.ndarray, res: int) -> np.ndarray:
    """Map pixel coordinates of a slice image to 4D positions.

    Parameters
    ----------
    o : np.ndarray
        Slice origin in 4D.
    a, b : np.ndarray
        Basis vectors spanning the slice plane.
    res : int
        Resolution of the output image (assumed square).

    Returns
    -------
    np.ndarray
        Array of shape ``(res, res, 4)`` containing the 4D positions of each
        pixel centre.
    """
    xs = np.linspace(-0.5, 0.5, res, endpoint=False, dtype=np.float32) + 0.5 / res
    ys = np.linspace(-0.5, 0.5, res, endpoint=False, dtype=np.float32) + 0.5 / res
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = o + grid_x[..., None] * a + grid_y[..., None] * b
    return points.astype(np.float32)


def eval_field(points: np.ndarray) -> np.ndarray:
    """Evaluate a simple CMYK-style field at 4D ``points``.

    Distances to the four canonical basis vectors are converted to CMYK weights
    via ``gelu`` and then mapped to RGB for visualisation.
    """
    centers = np.eye(4, dtype=np.float32)
    dists = np.linalg.norm(points[..., None, :] - centers[None, None, :, :], axis=-1)
    cmyk = gelu(1.0 - dists)
    rgb = 1.0 - cmyk[..., :3]
    return np.clip(rgb, 0.0, 1.0)


def temperature_from_margin(F_i: np.ndarray) -> float:
    """Compute a temperature from the score margin of a single pixel.

    Parameters
    ----------
    F_i:
        One-dimensional array of class scores for a pixel.

    Returns
    -------
    float
        Temperature ``tau_i = 1 + exp(-margin)`` where ``margin`` is the gap
        between the highest and second highest score in ``F_i``. A small margin
        therefore produces a high temperature and yields a softer softmax
        distribution.
    """

    sorted_scores = np.sort(F_i)
    margin = sorted_scores[-1] - sorted_scores[-2]
    return 1.0 + np.exp(-margin)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


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
    opacity_exp: float = 1.5,
) -> Dict[str, Any]:
    """Generate synthetic slices and return their file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = _field_density(res_coarse)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    origin_alpha = _field_density(res_hi)
    origin = np.dstack([origin_alpha] * 3)
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
    """Generate placeholder slices and basic rendering data.

    Besides writing placeholder images to ``output_dir`` this function now
    computes per-pixel class weights using a randomly initialized class loading
    matrix.  The returned dictionary therefore includes the generated paths as
    well as ``density`` and ``class_weights`` arrays for further processing.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density_path = out_dir / "coarse_density_map.png"
    origin_path = out_dir / "slice_origin.png"

    paths = {"origin": str(origin_path), "coarse_density": str(density_path)}

    # Generate rotated slices (placeholder using 90-degree rotations)
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

    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
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

    return {"paths": paths, "density": density, "class_weights": class_weights}


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
