import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from dashifine.palette import lineage_hue_from_address, eigen_palette


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


# -------------------------- neighbourhood statistics -------------------------

def estimate_sigma_knn(points4: np.ndarray, k: int) -> np.ndarray:
    """Estimate per-dimension scale from k-nearest neighbours.

    For each point in ``points4`` the ``k`` closest neighbours (excluding the
    point itself) are located.  The absolute difference along each axis is
    measured and the median across the neighbours is returned as ``sigma``.

    Parameters
    ----------
    points4:
        Array of shape ``(N, 4)`` containing the 4D coordinates of the points
        acting as centres.
    k:
        Number of neighbours to consider.  If fewer than ``k`` other points are
        available, all remaining points are used.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 4)`` where each row corresponds to the estimated
        ``sigma`` for the matching input point.
    """

    if points4.ndim != 2 or points4.shape[1] != 4:
        raise ValueError("points4 must be of shape (N, 4)")

    N = points4.shape[0]
    sigmas = np.zeros_like(points4, dtype=np.float32)

    for i in range(N):
        # Compute squared Euclidean distances to all other points
        diffs = points4 - points4[i]
        dists = np.sum(diffs ** 2, axis=1)

        # Exclude the point itself
        order = np.argsort(dists)
        order = order[order != i]
        k_eff = min(k, N - 1)
        if k_eff <= 0:
            continue

        neighbours = points4[order[:k_eff]]
        sigmas[i] = np.median(np.abs(neighbours - points4[i]), axis=0)

    return sigmas


# ----------------------------- density & classes -----------------------------

def alpha_eff(rho_tilde: np.ndarray, a_min: float = 0.6, a_max: float = 2.2, lam: float = 1.0, eta: float = 0.7) -> np.ndarray:
    t = np.clip(rho_tilde, 0.0, 1.0) ** eta
    return (1 - lam * t) * a_min + lam * t * a_max

def field_and_classes(points4: np.ndarray, centers: List[Dict[str, np.ndarray]], V: np.ndarray,
                      rho_eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate density and per-class weights for ``points4``.

    Parameters
    ----------
    points4:
        Array of shape ``(HW, 4)`` with slice sample positions.
    centers:
        List of dictionaries describing Gaussian-like centres with keys
        ``mu`` (4,), ``sigma`` (4,) and ``w`` (float).
    V:
        Class loadings of shape ``(C, N)`` mapping centre contributions to
        ``C`` classes.
    rho_eps:
        Small constant to avoid division by zero when normalising ``rho``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``rho`` of shape ``(HW,)`` and temperatured softmax weights ``W`` of
        shape ``(HW, C)`` ready for colouring.
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

    # Per-pixel temperature from top-2 margin and corresponding softmax
    W = np.zeros_like(F)
    for i in range(F.shape[0]):
        tau = temperature_from_margin(F[i])
        W[i] = softmax(F[i], tau=tau)

    return rho2, W


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

# ------------------------------- slice sampling ------------------------------

def sample_slice_points(H: int, W: int, origin4: np.ndarray, a4: np.ndarray, b4: np.ndarray, scale: float = 1.0) -> np.ndarray:
    u = np.linspace(-1, 1, W, dtype=np.float32)
    v = np.linspace(-1, 1, H, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    pts = origin4[None, :] + scale * (U.reshape(-1, 1) * a4[None, :] + V.reshape(-1, 1) * b4[None, :])
    return pts  # (HW,4)

def render_slice(
    H: int,
    W: int,
    origin4: np.ndarray,
    a4: np.ndarray,
    b4: np.ndarray,
    centers: List[Dict[str, np.ndarray]],
    V: np.ndarray,
    palette: str = "cmy",
    bg: Sequence[float] | np.ndarray = np.ones(3, dtype=np.float32),
    beta: float = 1.5,
) -> np.ndarray:
    """Render a single slice and composite against ``bg``.

    Parameters
    ----------
    H, W:
        Output image height and width.
    origin4, a4, b4:
        Slice origin and basis vectors in 4D.
    centers:
        List of centre dictionaries defining the field.
    V:
        Class loading matrix.
    palette:
        Colour mapping strategy (``"cmy"``, ``"eigen"``, ``"lineage"``).
    bg:
        RGB background colour to composite over.
    beta:
        Opacity exponent ``α = ρ̃^β``.

    Returns
    -------
    np.ndarray
        Composited RGB image of shape ``(H, W, 3)``.
    """

    pts = sample_slice_points(H, W, origin4, a4, b4, scale=1.0)
    rho, Wc = field_and_classes(pts, centers, V)

    C = Wc.shape[1]

    if palette.lower() == "cmy":
        CMY = np.zeros((Wc.shape[0], 3), dtype=np.float32)
        CMY[:, : min(3, C)] = np.clip(Wc[:, : min(3, C)], 0.0, 1.0)
        rgb = 1.0 - CMY
    elif palette.lower() == "eigen":
        rgb = eigen_palette(Wc)
    elif palette.lower() == "lineage":
        top_idx = np.argmax(Wc, axis=1)
        depth = np.max(Wc, axis=1)
        hsv = np.zeros((Wc.shape[0], 3), dtype=np.float32)
        for i, (idx, d) in enumerate(zip(top_idx, depth)):
            d_clip = np.clip(d, 0.0, 0.999)
            addr = f"{int(idx)}.{int(d_clip * 1000):03d}"
            h, s, v = lineage_hue_from_address(addr)
            hsv[i] = [h, s, v]
        rgb = hsv_to_rgb(hsv)
        if all("addr" in c for c in centers):
            centre_hsv = [lineage_hue_from_address(c["addr"]) for c in centers]
            centre_rgb = hsv_to_rgb(np.array(centre_hsv, dtype=np.float32))
            rgb = centre_rgb[top_idx]
    else:
        if C >= 2:
            CM = np.clip(Wc[:, :2], 0, 1)
            CMY = np.concatenate(
                [CM, np.zeros((Wc.shape[0], 1), dtype=np.float32)], axis=1
            )
            rgb = 1.0 - CMY
        else:
            rgb = np.repeat(np.max(Wc, axis=1).reshape(-1, 1), 3, axis=1)

    rgb = rgb.reshape(H, W, 3)
    alpha = opacity_from_density(rho, beta=beta).reshape(H, W, 1)
    bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    img = np.clip(rgb * alpha + bg_arr * (1.0 - alpha), 0.0, 1.0)
    return img


# ------------------------------------ main -----------------------------------

def main(
    output_dir: str | Path,
    res_hi: int = 128,
    res_coarse: int = 32,   # still used for a quick diagnostic map
    num_rotated: int = 4,
    num_time: int = 1,
    palette: str = "cmy",
    knn_k: int = 8,
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
    mus = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [-0.4, 0.8, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    weights = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    sigmas = estimate_sigma_knn(mus, knn_k)
    centers = []
    for mu, sigma, w in zip(mus, sigmas, weights):
        centers.append({"mu": mu, "sigma": sigma.astype(np.float32), "w": float(w)})
    V = np.eye(3, len(centers), dtype=np.float32)  # 3 classes from 3 centers

    paths: Dict[str, str] = {}
    for t in range(num_time):
        # increment the origin's w coordinate across normalised time [0,1]
        o_t = o.copy()
        o_t[3] = float(t) / max(num_time - 1, 1)

        # origin slice for this time step
        img0 = render_slice(
            res_hi, res_hi, o_t, a, b, centers, V, palette=palette
        )
        origin_path = out_dir / f"slice_t{t}_origin.png"
        plt.imsave(origin_path, img0)
        paths[f"t{t}_origin"] = str(origin_path)

        # rotated slices for this time step
        for i in range(num_rotated):
            angle = float(i) * 360.0 / max(num_rotated, 1)
            a_rot, b_rot = rotate_plane_4d(a, b, u, v, np.deg2rad(angle))
            img = render_slice(
                res_hi, res_hi, o_t, a_rot, b_rot, centers, V, palette=palette
            )
            rot_path = out_dir / f"slice_t{t}_rot_{int(angle):+d}deg.png"
            plt.imsave(rot_path, img)
            paths[f"t{t}_rot_{angle:+.1f}"] = str(rot_path)

    paths["coarse_density"] = str(density_path)
    return {"paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine slice renderer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=128)
    parser.add_argument("--res_coarse", type=int, default=32)
    parser.add_argument("--num_rotated", type=int, default=4)
    parser.add_argument("--num_time", type=int, default=1)
    parser.add_argument("--palette", type=str, default="cmy", choices=["cmy", "eigen", "lineage"])
    parser.add_argument("--knn_k", type=int, default=8, help="k for k-NN sigma estimation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        num_time=args.num_time,
        palette=args.palette,
        knn_k=args.knn_k,
    )
    # print(out)  # optional
