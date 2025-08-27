import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from typing import Tuple


def gelu(x: np.ndarray) -> np.ndarray:
    """Simple odd activation."""
    return np.tanh(x)


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Return orthonormal basis spanning ``a`` and ``b``."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b


def rotate_plane_4d(o: np.ndarray, a: np.ndarray, b: np.ndarray, u: np.ndarray, v: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``."""
    u, v = orthonormalize(u, v)
    ang = np.deg2rad(angle_deg)

    def _rot(x: np.ndarray) -> np.ndarray:
        xu, xv = np.dot(x, u), np.dot(x, v)
        x_perp = x - xu * u - xv * v
        xr = xu * np.cos(ang) - xv * np.sin(ang)
        yr = xu * np.sin(ang) + xv * np.cos(ang)
        return x_perp + xr * u + yr * v

    return _rot(o), _rot(a), _rot(b)


def rotate_plane(o: np.ndarray, a: np.ndarray, b: np.ndarray, axis: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatible wrapper using ``a`` and ``axis`` as rotation plane."""
    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


def p_adic_address_to_hue_saturation(addresses: np.ndarray, depth: np.ndarray, base: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Map p-adic addresses to hue and saturation components."""
    hue = (addresses % base).astype(np.float32) / float(base)
    sat = np.clip(depth, 0.0, 1.0).astype(np.float32)
    return hue, sat


def render(addresses: np.ndarray, depth: np.ndarray, palette: str = "p_adic", base: int = 4) -> np.ndarray:
    """Render RGB image for given ``addresses`` and ``depth``."""
    if palette != "p_adic":
        raise ValueError("Only p_adic palette supported in tests")
    hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base=base)
    hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
    return hsv_to_rgb(hsv)


def cmy_from_weights(W3: np.ndarray) -> np.ndarray:
    CMY = np.clip(W3, 0.0, 1.0)
    RGB = 1.0 - CMY
    return np.clip(RGB, 0.0, 1.0)


def opacity_from_density(rho: np.ndarray, beta: float = 1.0) -> np.ndarray:
    rho_t = rho / (np.max(rho) + 1e-6)
    return np.clip(rho_t ** beta, 0.0, 1.0)


def class_weights_to_rgba(weights: np.ndarray, density: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Compose class ``weights`` and ``density`` over white background."""
    rgb = cmy_from_weights(weights[..., :3])
    alpha = opacity_from_density(density, beta=beta)[..., None]
    return (1.0 - alpha) + alpha * rgb


def main(output_dir: str | Path, res_hi: int = 4, res_coarse: int = 2, num_rotated: int = 1,
         z0_steps: int = 1, w0_steps: int = 1, slopes: np.ndarray | None = None) -> dict:
    """Minimal stub creating a couple of placeholder images."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((res_hi, res_hi, 3), dtype=np.float32)
    slice_path = out_dir / "slice.png"
    plt.imsave(slice_path, img)
    coarse = np.zeros((res_coarse, res_coarse), dtype=np.float32)
    coarse_path = out_dir / "coarse.png"
    plt.imsave(coarse_path, coarse, cmap="gray")
    return {"paths": {"slice": str(slice_path), "coarse": str(coarse_path)}}


if __name__ == "__main__":
    main("examples")
