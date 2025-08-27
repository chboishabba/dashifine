import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import hashlib
import re


# ---------------------------- activations & utils -----------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    # Fast tanh-based GELU approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = (x - np.max(x)) / max(tau, 1e-8)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-8)

def temperature_from_margin(F: np.ndarray, tau_min: float = 0.25, tau_max: float = 2.5, gamma: float = 6.0) -> float:
    s = np.sort(F)[::-1]
    margin = s[0] - (s[1] if len(s) > 1 else 0.0)
    # Larger margin -> smaller temperature (crisper class)
    return tau_min + (tau_max - tau_min) / (1.0 + np.exp(gamma * margin))

def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b

def orthonormal_frame(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return orthonormalize(a, b)

def rotate_plane_4d(a: np.ndarray, b: np.ndarray, u: np.ndarray, v: np.ndarray, theta_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate the 2D slice basis (a,b) within 4D by angle theta around the 2D plane spanned by (u,v).
    u,v must be orthonormal 4D vectors.
    """
    def rot(vec: np.ndarray) -> np.ndarray:
        pu, pv = np.dot(vec, u), np.dot(vec, v)
        p = pu * u + pv * v
        q = vec - p
        p_rot = (pu * np.cos(theta_rad) - pv * np.sin(theta_rad)) * u + (pu * np.sin(theta_rad) + pv * np.cos(theta_rad)) * v
        return q + p_rot

    a2, b2 = rot(a), rot(b)
    return orthonormal_frame(a2, b2)


# ----------------------------- density & classes -----------------------------

def alpha_eff(rho_tilde: np.ndarray, a_min: float = 0.6, a_max: float = 2.2, lam: float = 1.0, eta: float = 0.7) -> np.ndarray:
    t = np.clip(rho_tilde, 0.0, 1.0) ** eta
    return (1 - lam * t) * a_min + lam * t * a_max

def field_and_classes(points4: np.ndarray, centers: List[Dict[str, np.ndarray]], V: np.ndarray,
                      rho_eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    points4: (HW,4)
    centers: list of { 'mu':(4,), 'sigma':(4,), 'w':float }
    V: (C,N) class loadings (C classes, N centers)
    returns: rho (HW,), F (HW,C)
    """
    HW = points4.shape[0]
    N = len(centers)
    g = np.zeros((HW, N), dtype=np.float32)

    # Pass 1: provisional alpha=1 for rho_tilde estimate
    for j, c in enumerate(centers):
        r = np.linalg.norm((points4 - c["mu"]) / (c["sigma"] + 1e-8), axis=1)
        g[:, j] = c["w"] * gelu(1.0 - r)

    rho = np.sum(g, axis=1)
    rho_tilde = rho / (np.max(rho) + rho_eps)

    # Pass 2: mass-coupled sharpness α_eff(ρ̃)
    g2 = np.zeros_like(g)
    aeff = alpha_eff(rho_tilde)  # (HW,)
    for j, c in enumerate(centers):
        r = np.linalg.norm((points4 - c["mu"]) / (c["sigma"] + 1e-8), axis=1)
        g2[:, j] = c["w"] * gelu(aeff * (1.0 - r))

    F = g2 @ V.T  # (HW,C)
    rho2 = np.sum(g2, axis=1)
    return rho2, F


# ------------------------------- colour mapping ------------------------------

def cmy_from_weights(W3: np.ndarray) -> np.ndarray:
    """
    W3: (HW,3) in [0,1], sum ~ 1
    Returns RGB in [0,1] via RGB = 1 - CMY.
    """
    CMY = np.clip(W3, 0.0, 1.0)
    RGB = 1.0 - CMY
    return np.clip(RGB, 0.0, 1.0)

def opacity_from_density(rho: np.ndarray, beta: float = 1.5) -> np.ndarray:
    rho_t = rho / (np.max(rho) + 1e-6)
    return np.clip(np.power(rho_t, beta), 0.0, 1.0)

# ---- palette hooks you can extend -------------------------------------------

def lineage_hue_from_address(addr_digits: str, base: int = 4) -> Tuple[float, float, float]:
    """Map a p-adic style address string to HSV components.

    Parameters
    ----------
    addr_digits:
        Address string. An optional fractional part encodes depth.  The
        integer portion is split into a *prefix* (supervoxel) and an integer
        suffix which is interpreted as base-``p`` digits.
    base:
        Base ``p`` used to interpret the integer suffix.

    Returns
    -------
    tuple[float, float, float]
        ``(h, s, v)`` in the range ``[0, 1]``.
    """

    # Separate fractional depth
    if "." in addr_digits:
        addr_main, frac_part = addr_digits.split(".", 1)
    else:
        addr_main, frac_part = addr_digits, ""

    # Split prefix (supervoxel) and integer suffix
    m = re.match(r"(\d*?)(\d+)$", addr_main)
    if m:
        prefix_digits, suffix_digits = m.group(1), m.group(2)
    else:
        prefix_digits, suffix_digits = "", addr_main

    # Stable hue from prefix digits via SHA256 hash
    if prefix_digits:
        h = hashlib.sha256(prefix_digits.encode("utf-8")).hexdigest()
        prefix_hue = int(h[:8], 16) / 0xFFFFFFFF
    else:
        prefix_hue = 0.0

    # Interpret suffix digits as base-p digits contributing fractional hue
    hue = prefix_hue
    for k, ch in enumerate(reversed(suffix_digits)):
        digit = min(int(ch), base - 1)
        hue += digit / (base ** (k + 1))
    hue = hue % 1.0

    # Fractional depth controls saturation/value
    depth = float(f"0.{frac_part}") if frac_part else 0.0
    saturation = np.clip(depth, 0.0, 1.0)
    value = 1.0 - 0.5 * depth

    return float(hue), float(saturation), float(value)

def eigen_palette(W: np.ndarray) -> np.ndarray:
    """
    TODO: project class weights to 3D (PCA/UMAP/etc.) for RGB; stub uses top-class gray.
    W: (HW,C)
    """
    top = np.max(W, axis=1, keepdims=True)
    RGB = np.repeat(top, 3, axis=1)
    return np.clip(RGB, 0.0, 1.0)


# ------------------------------- slice sampling ------------------------------

def sample_slice_points(H: int, W: int, origin4: np.ndarray, a4: np.ndarray, b4: np.ndarray, scale: float = 1.0) -> np.ndarray:
    u = np.linspace(-1, 1, W, dtype=np.float32)
    v = np.linspace(-1, 1, H, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    pts = origin4[None, :] + scale * (U.reshape(-1, 1) * a4[None, :] + V.reshape(-1, 1) * b4[None, :])
    return pts  # (HW,4)

def render_slice(H: int, W: int, origin4: np.ndarray, a4: np.ndarray, b4: np.ndarray,
                 centers: List[Dict[str, np.ndarray]], V: np.ndarray, palette: str = "cmy") -> Tuple[np.ndarray, np.ndarray]:
    pts = sample_slice_points(H, W, origin4, a4, b4, scale=1.0)
    rho, F = field_and_classes(pts, centers, V)

    # temperatured softmax per pixel
    C = F.shape[1]
    Wc = np.zeros_like(F)
    for i in range(F.shape[0]):
        tau = temperature_from_margin(F[i])
        Wc[i] = softmax(F[i], tau=tau)

    if palette.lower() == "cmy" and C >= 3:
        RGB = cmy_from_weights(Wc[:, :3]).reshape(H, W, 3)
    elif palette.lower() == "eigen":
        RGB = eigen_palette(Wc).reshape(H, W, 3)
    elif palette.lower() == "lineage":
        top_idx = np.argmax(Wc, axis=1)
        depth = np.max(Wc, axis=1)
        hsv = np.zeros((Wc.shape[0], 3), dtype=np.float32)
        for i, (idx, d) in enumerate(zip(top_idx, depth)):
            d_clip = np.clip(d, 0.0, 0.999)
            addr = f"{int(idx)}.{int(d_clip * 1000):03d}"
            h, s, v = lineage_hue_from_address(addr)
            hsv[i] = [h, s, v]
        RGB = hsv_to_rgb(hsv).reshape(H, W, 3)
    else:
        # 2-class CM (Cyan/Magenta) or generic grayscale fallback
        if C >= 2:
            CM = np.clip(Wc[:, :2], 0, 1)  # [C,M]
            # fill Y=0, make 3 channels CMY -> RGB
            CMY = np.concatenate([CM, np.zeros((Wc.shape[0], 1), dtype=np.float32)], axis=1)
            RGB = (1.0 - CMY).reshape(H, W, 3)
        else:
            RGB = np.repeat(np.max(Wc, axis=1).reshape(H, W, 1), 3, axis=2)

    A = opacity_from_density(rho).reshape(H, W, 1)
    return np.clip(RGB, 0, 1), A


# ------------------------------------ main -----------------------------------

def main(
    output_dir: str | Path,
    res_hi: int = 128,
    res_coarse: int = 32,   # still used for a quick diagnostic map
    num_rotated: int = 4,
    palette: str = "cmy",
) -> Dict[str, Any]:
    """Render actual Dashifine slices and return file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # quick coarse density placeholder for continuity
    density = np.zeros((res_coarse, res_coarse), dtype=np.float32)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    # --- define a simple 4D slice basis & rotation plane
    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    a, b = orthonormal_frame(a, b)
    u = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    u, v = orthonormal_frame(u, v)

    # --- tiny demo scene: 3 centers, 3 classes (CMY) -------------------------
    centers = [
        {"mu": np.array([0.0, 0.0, 0.0, 0.0], np.float32), "sigma": np.array([0.7, 0.7, 0.7, 0.7], np.float32), "w": 1.0},
        {"mu": np.array([0.8, 0.2, 0.0, 0.0], np.float32), "sigma": np.array([0.5, 0.7, 0.7, 0.7], np.float32), "w": 0.9},
        {"mu": np.array([-0.4, 0.8, 0.0, 0.0], np.float32), "sigma": np.array([0.7, 0.5, 0.7, 0.7], np.float32), "w": 0.8},
    ]
    V = np.eye(3, len(centers), dtype=np.float32)  # 3 classes from 3 centers

    paths: Dict[str, str] = {}
    # origin slice (no rotation)
    img0, A0 = render_slice(res_hi, res_hi, o, a, b, centers, V, palette=palette)
    rgba0 = np.clip(np.dstack([img0, A0]), 0, 1)
    origin_path = out_dir / "slice_origin.png"
    plt.imsave(origin_path, rgba0)
    paths["origin"] = str(origin_path)

    # rotated slices
    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        a_rot, b_rot = rotate_plane_4d(a, b, u, v, np.deg2rad(angle))
        img, A = render_slice(res_hi, res_hi, o, a_rot, b_rot, centers, V, palette=palette)
        rgba = np.clip(np.dstack([img, A]), 0, 1)
        rot_path = out_dir / f"slice_rot_{int(angle):+d}deg.png"
        plt.imsave(rot_path, rgba)
        paths[f"rot_{angle:+.1f}"] = str(rot_path)

    paths["coarse_density"] = str(density_path)
    return {"paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine slice renderer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=128)
    parser.add_argument("--res_coarse", type=int, default=32)
    parser.add_argument("--num_rotated", type=int, default=4)
    parser.add_argument("--palette", type=str, default="cmy", choices=["cmy", "eigen", "lineage"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        palette=args.palette,
    )
    # print(out)  # optional
