import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from typing import Tuple
"""Dashifine demo utilities.

This module contains a tiny synthetic 4D field along with a number of helper
functions used throughout the tests.  The implementation is intentionally small
and is not meant to be a feature complete renderer – many of the operations are
simple placeholders that nevertheless exercise the control flow of the real
project.

The script can also be executed directly to generate a couple of example images
showcasing the different colour palettes.
"""

"""Dashifine slice renderer with simple 4‑D geometry utilities.

This module exposes a small set of functions that are exercised by the unit
tests.  Only a toy procedural field is implemented – enough to verify that the
geometry helpers behave correctly and that successive rotations lead to
distinct images.
"""

"""Core rendering utilities for the Dashifine demos.

This module offers a grab‑bag of small helpers used by the tests.  It contains
basic maths primitives, simple colour utilities and a tiny demo ``main``
function capable of rendering a few rotated 4‑D slices.  The implementation is
deliberately compact; it is not intended to be a full featured renderer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from dataclasses import dataclass, field


@dataclass
class FieldCenters:
    """Minimal container for synthetic field parameters.

    This lightweight placeholder ensures the module imports during tests without
    requiring the full demo configuration."""

    mu: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    sigma: np.ndarray = field(
        default_factory=lambda: np.ones((0, 2), dtype=np.float32)
    )
    w: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=np.float32))


BETA: float = 1.5
CENTERS: FieldCenters = FieldCenters()


# ---------------------------------------------------------------------------
# Basic maths helpers


# ---------------------------------------------------------------------------
# Basic data structures
# ---------------------------------------------------------------------------


@dataclass
class FieldCenters:
    """Parameterisation of anisotropic radial basis functions in 4D."""

    mu: np.ndarray
    """Centre positions with shape ``(N, 4)``."""

    sigma: np.ndarray
    """Per-axis standard deviations for anisotropic falloff, shape ``(N, 4)``."""


def gelu(x: np.ndarray) -> np.ndarray:
    """Simple odd activation used in tests."""

    return np.tanh(x)


# ---------------------------------------------------------------------------
# basic primitives
# ---------------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """Tiny odd activation used in the tests."""

=======

@dataclass
class FieldCenters:
    """Container for centre parameters ``mu``, ``sigma`` and ``w``."""
    mu: np.ndarray
    sigma: np.ndarray
    w: np.ndarray


BETA: float = 1.5
CENTERS = FieldCenters(
    mu=np.zeros((1, 2), dtype=np.float32),
    sigma=np.ones((1, 2), dtype=np.float32),
    w=np.ones(1, dtype=np.float32),
)

# A few hard coded centres used by tests
# Default centres used for examples and tests.  The ``z`` and ``w`` coordinates
# are zero so that :func:`_field_density` can project them to the ``x``/``y``
# plane without additional parameters.
CENTERS = FieldCenters(
    mu=np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [-0.5, 0.25, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
    sigma=np.array(
        [
            [0.3, 0.3, 0.3, 0.3],
            [0.25, 0.25, 0.25, 0.25],
            [0.35, 0.35, 0.35, 0.35],
            [0.6, 0.6, 0.6, 0.6],
            [0.4, 0.7, 0.6, 0.6],
            [0.6, 0.4, 0.6, 0.6],
        ],
        dtype=np.float32,
    ),
    w=np.array([1.0, 0.8, 0.9], dtype=np.float32),
)

# Exponent used when converting density to opacity
BETA = 1.5


# ---------------------------------------------------------------------------
# Small maths helpers
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""

    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def temperature_from_margin(F_i: np.ndarray) -> float:
    """Temperature schedule used for class weighting."""

    sorted_scores = np.sort(F_i)
    margin = sorted_scores[-1] - sorted_scores[-2]
    return 1.0 + np.exp(-margin)
# ------------------------------ basic primitives -----------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """Simple odd activation."""
    """Light‑weight GELU approximation used in a few tests."""

    """Simple odd activation used in tests."""
    return np.tanh(x)


def field_and_classes(
    points4: np.ndarray,
    centers: FieldCenters,
    V: np.ndarray,
    rho_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate density and class scores for 4D ``points4``.

    Parameters
    ----------
    points4:
        Array of shape ``(HW, 4)`` containing 4D sample positions.
    centers:
        ``FieldCenters`` describing kernel centres, anisotropy and weights.
    V:
        Class loading matrix of shape ``(C, N)``.
    rho_eps:
        Small constant to avoid division by zero when normalising ``rho``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Total density ``rho`` with shape ``(HW,)`` and class scores ``F`` with
        shape ``(HW, C)``.
    """

    mu, sigma, w = centers.mu, centers.sigma, centers.w

    # Anisotropic distances r_i = ||(p - mu_i) / sigma_i||
    diff = points4[:, None, :] - mu[None, :, :]  # (HW, N, 4)
    ri = np.linalg.norm(diff / (sigma[None, :, :] + 1e-8), axis=-1)  # (HW, N)
def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Return orthonormal basis spanning ``a`` and ``b``."""
    """Orthonormalise vectors ``a`` and ``b`` using Gram–Schmidt."""


    """Orthonormalise ``a`` and ``b`` using Gram–Schmidt."""

    """Orthonormalise vectors ``a`` and ``b`` with Gram–Schmidt."""

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b


# ---------------------------------------------------------------------------
# rotation helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 4‑D rotation and sampling utilities
def rotate_plane(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``(o, a, b)`` around ``axis`` using :func:`rotate_plane_4d`."""

    # Pass 1: provisional alpha=1 to estimate rho_tilde
    g = w[None, :] * gelu(1.0 - ri)
    rho = np.sum(g, axis=1)
    rho_tilde = rho / (np.max(rho) + rho_eps)

    # Pass 2: mass-coupled sharpness via alpha_eff(rho_tilde)
    alpha_eff = 1.0 / (1.0 + rho_tilde)  # (HW,)
    g = w[None, :] * gelu(alpha_eff[:, None] * (1.0 - ri))

    rho = np.sum(g, axis=1)
    F = g @ V.T
    return rho, F


def orthonormalize(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalize vectors ``a`` and ``b`` with Gram-Schmidt."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + eps)
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + eps)
    return a, b
def rotate_plane_4d(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``.

    Any component of the inputs lying in this plane is rotated by
    ``angle_deg`` degrees while the orthogonal component is left unchanged.
    """

    The plane is defined by two (not necessarily normalised) vectors ``u`` and
    ``v``. Any component of the inputs lying in this plane is rotated by
    ``angle_deg`` degrees while the orthogonal component is left unchanged.
    """
    u, v = orthonormalize(u, v)
    theta = np.deg2rad(angle_deg)
    """
    ``u`` and ``v`` need not be normalised; they simply define the rotation
    plane.  Components of the inputs that lie in this plane are rotated by
    ``angle_deg`` degrees while orthogonal components remain unchanged.
    """

def rotate_plane_4d(o: np.ndarray, a: np.ndarray, b: np.ndarray, u: np.ndarray, v: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ``o``, ``a`` and ``b`` in the plane spanned by ``u`` and ``v``."""
    u, v = orthonormalize(u, v)
    ang = np.deg2rad(angle_deg)

    def _rotate(x: np.ndarray) -> np.ndarray:
        xu = float(np.dot(x, u))
        xv = float(np.dot(x, v))
    def _rot(x: np.ndarray) -> np.ndarray:
        xu, xv = np.dot(x, u), np.dot(x, v)
        x_perp = x - xu * u - xv * v
        xu = np.dot(x, u)
        xv = np.dot(x, v)
        x_perp = x - xu * u - xv * v
        xr = xu * np.cos(theta) - xv * np.sin(theta)
        yr = xu * np.sin(theta) + xv * np.cos(theta)
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




def rotate_plane(
    o: np.ndarray, a: np.ndarray, b: np.ndarray, axis: np.ndarray, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward‑compatible wrapper using ``a`` and ``axis`` as rotation plane."""

    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


# ---------------------------------------------------------------------------
# Slice sampling and simple field evaluation
# ---------------------------------------------------------------------------
    """Backward compatible wrapper around :func:`rotate_plane_4d`.

    The previous API expected a single rotation ``axis``.  We use ``a`` together
    with ``axis`` to define the rotation plane and delegate to
    :func:`rotate_plane_4d`.
    """

    return rotate_plane_4d(o, a, b, a, axis, angle_deg)


def sample_slice_points(
    H: int, W: int, origin4: np.ndarray, a4: np.ndarray, b4: np.ndarray
) -> np.ndarray:
    """Map a ``H×W`` pixel grid to 4‑D positions.

    The slice is centred on ``origin4`` with basis vectors ``a4`` and ``b4``
    spanning the pixel grid in the range ``[-1, 1]`` along each axis.
    """

    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = (
        origin4[None, :]
        + grid_x.reshape(-1, 1) * a4[None, :]
        + grid_y.reshape(-1, 1) * b4[None, :]
    )
    return pts


# ---------------------------------------------------------------------------
# Field evaluation


@dataclass
class Center:
    mu: np.ndarray
    sigma: np.ndarray
    w: float

def sample_slice_image(o: np.ndarray, a: np.ndarray, b: np.ndarray, res: int) -> np.ndarray:
    """Map pixel coordinates of a slice image to 4D positions."""

    xs = np.linspace(-0.5, 0.5, res, endpoint=False, dtype=np.float32) + 0.5 / res
    ys = np.linspace(-0.5, 0.5, res, endpoint=False, dtype=np.float32) + 0.5 / res
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = o + grid_x[..., None] * a + grid_y[..., None] * b
    return points.astype(np.float32)

def alpha_eff(
    rho_tilde: np.ndarray, a_min: float = 0.6, a_max: float = 2.2, lam: float = 1.0, eta: float = 0.7
) -> np.ndarray:
    t = np.clip(rho_tilde, 0.0, 1.0) ** eta
    return (1 - lam * t) * a_min + lam * t * a_max


def field_and_classes(
    points4: np.ndarray, centers: Iterable[Center], V: np.ndarray, rho_eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the toy field and class scores at 4‑D positions."""

def eval_field(points: np.ndarray) -> np.ndarray:
    """Evaluate a simple CMYK‑style field at 4D ``points``."""

    centers = np.eye(4, dtype=np.float32)
    dists = np.linalg.norm(points[..., None, :] - centers[None, None, :, :], axis=-1)
    cmyk = gelu(1.0 - dists)
    rgb = 1.0 - cmyk[..., :3]
    return np.clip(rgb, 0.0, 1.0)
    pts = points4.astype(np.float32)
    centers_list = list(centers)
    N = len(centers_list)
    HW = pts.shape[0]

    g = np.zeros((HW, N), dtype=np.float32)
    for j, c in enumerate(centers_list):
        r = np.linalg.norm((pts - c.mu) / (c.sigma + 1e-8), axis=1)
        g[:, j] = c.w * gelu(1.0 - r)
    g = np.maximum(g, 0.0)

    rho = np.sum(g, axis=1)
    rho_tilde = rho / (np.max(rho) + rho_eps)

def temperature_from_margin(F_i: np.ndarray) -> float:
    """Compute a softmax temperature from the score margin of a pixel."""
    g2 = np.zeros_like(g)
    a_eff = alpha_eff(rho_tilde)
    for j, c in enumerate(centers_list):
        r = np.linalg.norm((pts - c.mu) / (c.sigma + 1e-8), axis=1)
        g2[:, j] = c.w * gelu(a_eff * (1.0 - r))
    g2 = np.maximum(g2, 0.0)

def rotate_plane(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    F = g2 @ V.T
    rho_final = np.sum(g2, axis=1)
    return rho_final, F


# ---------------------------------------------------------------------------
# Colour utilities



def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)
    """Backward compatible wrapper around :func:`rotate_plane_4d`."""

def rotate_plane(o: np.ndarray, a: np.ndarray, b: np.ndarray, axis: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatible wrapper for :func:`rotate_plane_4d`."""

    return rotate_plane_4d(o, a, b, a, axis, angle_deg)

def mix_cmy_to_rgb(weights: np.ndarray) -> np.ndarray:
    """Mix CMY(K) weights to RGB."""


# ----------------------------- density & classes -----------------------------

def alpha_eff(
    rho_tilde: np.ndarray,
    a_min: float = 0.6,
    a_max: float = 2.2,
    lam: float = 1.0,
    eta: float = 0.7,
) -> np.ndarray:
    """Effective sharpness ``α_eff(ρ̃)`` used to couple mass and kernel width."""
    t = np.clip(rho_tilde, 0.0, 1.0) ** eta
    return (1 - lam * t) * a_min + lam * t * a_max


def field_and_classes(
    points4: np.ndarray,
    centers: List[Dict[str, np.ndarray]],
    V: np.ndarray,
    rho_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute density and class scores for ``points4``.

    Parameters
    ----------
    points4:
        Array of shape ``(HW, 4)`` containing sample points.
    centers:
        List of dictionaries with ``mu``, ``sigma`` and ``w`` for each centre.
    V:
        Array of shape ``(C, N)`` giving class loadings.
    rho_eps:
        Small constant to avoid division by zero in normalisation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``rho`` of shape ``(HW,)`` and class scores ``F`` of shape ``(HW, C)``.
    """
    HW = points4.shape[0]
    N = len(centers)
    g = np.zeros((HW, N), dtype=np.float32)

    # Pass 1: provisional kernels with α = 1 to estimate ρ̃
    for j, c in enumerate(centers):
        r = np.linalg.norm((points4 - c["mu"]) / (c["sigma"] + 1e-8), axis=1)
        g[:, j] = c["w"] * gelu(1.0 - r)

    rho = np.sum(g, axis=1)
    rho_tilde = rho / (np.max(rho) + rho_eps)

    # Pass 2: re-evaluate kernels with α_eff(ρ̃)
    g2 = np.zeros_like(g)
    aeff = alpha_eff(rho_tilde)
    for j, c in enumerate(centers):
        r = np.linalg.norm((points4 - c["mu"]) / (c["sigma"] + 1e-8), axis=1)
        g2[:, j] = c["w"] * gelu(aeff * (1.0 - r))

    F = g2 @ V.T
    rho = np.sum(g2, axis=1)
    return rho, F

def sample_slice_image(H: int, W: int, origin4: np.ndarray, a4: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """Map each pixel of an ``H``×``W`` grid to 4‑D coordinates.

    The slice is defined by ``origin4`` and basis vectors ``a4`` and ``b4``. For
    pixel coordinates ``(u, v)`` in ``[-1, 1]`` the mapped point is

    ``x = origin4 + u * a4 + v * b4``.
    """

    u = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="xy")
    pts = origin4[None, None, :] + U[..., None] * a4[None, None, :] + V[..., None] * b4[None, None, :]
    return pts


# ---------------------------------------------------------------------------
# colour utilities
# ---------------------------------------------------------------------------

def mix_cmy_to_rgb(weights: np.ndarray) -> np.ndarray:
    cmy = np.clip(weights[..., :3], 0.0, 1.0)
    k = np.clip(weights[..., 3:4], 0.0, 1.0)
    rgb = (1.0 - cmy) * (1.0 - k)
    return np.clip(rgb, 0.0, 1.0)


def density_to_alpha(density: np.ndarray, beta: float = BETA) -> np.ndarray:
def density_to_alpha(density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    density = np.clip(density, 0.0, 1.0)
    return np.power(density, beta)


def composite_rgb_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Composite an RGB image against ``bg`` using the supplied alpha."""

    rgb: np.ndarray, alpha: np.ndarray, bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:

def composite_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray, bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.float32)
    return rgb * alpha[..., None] + bg_arr * (1.0 - alpha[..., None])


@dataclass
class FieldCenters:
    """Simple container for field centre parameters."""

    mu: np.ndarray
    sigma: np.ndarray
    w: np.ndarray


CENTERS = FieldCenters(
    mu=np.zeros((1, 2), dtype=np.float32),
    sigma=np.ones((1, 2), dtype=np.float32),
    w=np.ones(1, dtype=np.float32),
)

BETA = 1.5

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------


def lineage_hue_from_address(addr: str) -> float:
    """Return a deterministic hue in ``[0, 1]`` for ``addr``.

    The current implementation hashes the address string and uses the hash to
    generate a stable hue.  The exact mapping is unimportant for the tests – it
    merely needs to be deterministic.
    """

    h = hash(addr) & 0xFFFFFFFF
    return (h / 0xFFFFFFFF) % 1.0


def lineage_hsv_from_address(addr_digits: str, base: int = 4) -> Tuple[float, float, float]:
    """Map a p-adic style address string to HSV components."""

    import hashlib
    import re

    if "." in addr_digits:
        addr_main, frac_part = addr_digits.split(".", 1)
    else:
        addr_main, frac_part = addr_digits, ""

    m = re.match(r"(\d*?)(\d+)$", addr_main)
    if m:
        prefix_digits, suffix_digits = m.group(1), m.group(2)
    else:
        prefix_digits, suffix_digits = "", addr_main

    if prefix_digits:
        h = hashlib.sha256(prefix_digits.encode("utf-8")).hexdigest()
        prefix_hue = int(h[:8], 16) / 0xFFFFFFFF
    else:
        prefix_hue = 0.0

    hue = prefix_hue
    for k, ch in enumerate(reversed(suffix_digits)):
        digit = min(int(ch), base - 1)
        hue += digit / (base ** (k + 1))
    hue %= 1.0

    depth = float(f"0.{frac_part}") if frac_part else 0.0
    saturation = np.clip(depth, 0.0, 1.0)
    value = 1.0 - 0.5 * depth
    return float(hue), float(saturation), float(value)


def eigen_palette(weights: np.ndarray) -> np.ndarray:
    """Placeholder eigen palette mapping to grayscale."""

    g = np.mean(weights, axis=-1, keepdims=True)
    return np.repeat(g, 3, axis=-1)
def eigen_palette(W: np.ndarray) -> np.ndarray:
    """Project class weights to their first three principal components."""

    if W.ndim == 3:
        flat = W.reshape(-1, W.shape[-1])
    else:
        flat = W
    if flat.size == 0:
        return np.zeros((flat.shape[0], 3), dtype=np.float32)

    Wc = flat - np.mean(flat, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
    proj = Wc @ Vt[:3].T
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])), mode="constant")
    mn = proj.min(axis=0, keepdims=True)
    mx = proj.max(axis=0, keepdims=True)
    denom = np.where(mx - mn > 1e-8, mx - mn, 1.0)
    rgb = (proj - mn) / denom
    return np.clip(rgb, 0.0, 1.0)


def class_weights_to_rgba(
    class_weights: np.ndarray,
    density: np.ndarray,
    beta: float = BETA,
) -> np.ndarray:
    """Convert class weights to an RGB image composited on white."""
    class_weights: np.ndarray, density: np.ndarray, beta: float = 1.5
) -> np.ndarray:

    Up to the first three channels of ``class_weights`` are interpreted as
    cyan, magenta and yellow contributions. Missing channels are assumed to be
    zero, yielding a CMY triplet which is converted to RGB using ``RGB = 1 -
    CMY``. Opacity is computed as ``density ** beta`` and the RGB image is
    composited over a white background.

    Parameters
    ----------
    class_weights:
        Array of shape ``(H, W, C)`` with ``C >= 1`` containing per-class
        weights. Up to the first three channels are used as CMY components; any
        remaining channels are ignored.
    density:
        Array of shape ``(H, W)`` giving normalised density ``rho_tilde``.
    beta:
        Exponent controlling opacity from density.
    """Map class weights and density to a composited RGB image."""

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
    Returns
    -------
    np.ndarray
        Composited RGB image in ``[0, 1]``.
    """

    cmy = np.zeros(class_weights.shape[:2] + (3,), dtype=class_weights.dtype)
    channels = min(class_weights.shape[-1], 3)
    cmy[..., :channels] = class_weights[..., :channels]
    rgb = 1.0 - np.clip(cmy, 0.0, 1.0)


def class_weights_to_rgba(class_weights: np.ndarray, density: np.ndarray, beta: float = 1.5) -> np.ndarray:
    """Map class weights and density to a composited RGB image."""

    k = np.zeros(class_weights.shape[:2] + (1,), dtype=class_weights.dtype)
    cmyk = np.concatenate([class_weights[..., :3], k], axis=-1)
    rgb = mix_cmy_to_rgb(cmyk)
    alpha = density_to_alpha(density, beta)
    return composite_rgb_alpha(rgb, alpha)


# ---------------------------------------------------------------------------
# p-adic visualisation utilities
# ---------------------------------------------------------------------------
# P‑adic helper used by a couple of tests
# ---------------------------------------------------------------------------



def p_adic_address_to_hue_saturation(
    addresses: np.ndarray,
    depth: np.ndarray,
    base: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map p‑adic addresses to hue and depth to saturation."""

    """Map p-adic addresses to hue and depth to saturation."""



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

    max_power = int(np.ceil(np.log(addresses.max() + 1) / np.log(base))) if np.any(addresses) else 1
    hue = np.zeros_like(addresses, dtype=np.float32)
    for k in range(max_power):
        digit = (addresses // (base ** k)) % base
        hue += digit / (base ** (k + 1))

    saturation = depth / (np.max(depth) + 1e-8) if np.any(depth) else np.zeros_like(depth)
    return hue, saturation


def render(
    addresses: np.ndarray,
    depth: np.ndarray,
    *,
    palette: str = "gray",
    base: int = 2,
) -> np.ndarray:
    """Render an RGB image from ``addresses`` and ``depth``."""

def render(addresses: np.ndarray, depth: np.ndarray, *, palette: str = "gray", base: int = 2) -> np.ndarray:
    """Render an RGB image from ``addresses`` and ``depth``."""

    if palette == "p_adic":
        hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base=base)
        hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
        return hsv_to_rgb(hsv)

    # default grayscale based on normalised depth
    depth_n = depth / (np.max(depth) + 1e-8) if np.any(depth) else depth
    return np.stack([depth_n] * 3, axis=-1)


# ---------------------------------------------------------------------------
# tiny demo renderer
# ---------------------------------------------------------------------------

def eval_field(points4: np.ndarray) -> np.ndarray:
    """Simple 4-D field used for demo rendering.

    The field varies along the ``z`` and ``w`` axes which makes rotations in
    those dimensions visually apparent.
    """

    z = points4[..., 2]
    w = points4[..., 3]
    val = np.sin(3.0 * z) + np.cos(3.0 * w)
    val = (val - val.min()) / (val.max() - val.min() + 1e-8)
    return np.stack([val, val, val], axis=-1)
# Slice rendering with palette selection
# ---------------------------------------------------------------------------


def render_slice(
    H: int,
    W: int,
    origin4: np.ndarray,
    a4: np.ndarray,
    b4: np.ndarray,
    centers: Any,
    V: np.ndarray,
    palette: str = "cmy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a coloured slice using one of several palettes.

    The implementation is intentionally simple: the returned colours do not
    attempt to represent the underlying field faithfully but they allow the
    tests to exercise the palette selection logic.
    """

    x = np.linspace(0.0, 1.0, W, dtype=np.float32)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    weights = np.stack([X, Y, 1.0 - X, 0.5 * np.ones_like(X)], axis=-1)

    if palette.lower() == "eigen":
        rgb = eigen_palette(weights).reshape(H, W, 3)
    elif palette.lower() == "lineage":
        top = np.argmax(weights, axis=-1)
        hsv = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                hue = lineage_hue_from_address(str(int(top[i, j])))
                hsv[i, j] = [hue, 1.0, 1.0]
        rgb = hsv_to_rgb(hsv)
    else:  # default CMY
        rgb = mix_cmy_to_rgb(weights)

    density = weights.mean(axis=-1)
    alpha = density_to_alpha(density, BETA)
    return rgb, alpha


# ---------------------------------------------------------------------------
# Minimal field density and main demo entry point
# ---------------------------------------------------------------------------


def _field_density(res: int, centers: FieldCenters = CENTERS, beta: float = BETA) -> np.ndarray:
    """Evaluate the synthetic field on a ``res``×``res`` grid."""

    mu, sigma, w = centers.mu, centers.sigma, centers.w

    lin = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing="xy")
    pos = np.stack([X, Y, np.zeros_like(X), np.zeros_like(X)], axis=-1)

    diff = pos[None, ...] - mu[:, None, None, :]
    ri = np.linalg.norm(diff / sigma[:, None, None, :], axis=-1)

    g = w[:, None, None] * gelu(1.0 - ri)
    rho = g.sum(axis=0)
    rho_tilde = (rho - rho.min()) / (rho.max() - rho.min() + 1e-8)
    alpha_vis = rho_tilde ** beta
    return alpha_vis
# Rendering pipeline


def _demo_centers() -> Tuple[List[Center], np.ndarray]:
    mus = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.5, 0.0],
            [-0.4, 0.8, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    sigmas = np.full_like(mus, 1.0, dtype=np.float32)
    weights = np.array([1.0, 0.9, 0.8], dtype=np.float32)

    centers: List[Center] = []
    for mu, sigma, w in zip(mus, sigmas, weights):
        centers.append(Center(mu=mu, sigma=sigma, w=float(w)))
    # Compute anisotropic distances r_i for each centre
    diff = pos[None, ...] - mu[:, None, None, :2]  # (N, res, res, 2)
    ri = np.linalg.norm(diff / sigma[:, None, None, :2], axis=-1)  # (N, res, res)

    V = np.eye(3, len(centers), dtype=np.float32)  # map centres to CMY
    return centers, V


def render_slice(
    res: int,
    origin4: np.ndarray,
    a4: np.ndarray,
    b4: np.ndarray,
    centers: List[Center],
    V: np.ndarray,
) -> np.ndarray:
    pts = sample_slice_points(res, res, origin4, a4, b4)
    rho, F = field_and_classes(pts, centers, V)
    weights = np.zeros_like(F)
    for i in range(F.shape[0]):
        tau = temperature_from_margin(F[i])
        weights[i] = softmax(F[i] / tau, axis=0)
    rgb = class_weights_to_rgba(
        weights.reshape(res, res, -1), rho.reshape(res, res)
    )
    return rgb


def main(
    output_dir: str | Path,
    res_hi: int = 64,
    num_rotated: int = 1,
    **_: Any,
) -> Dict[str, Any]:
    """Render an origin slice and a number of rotated slices."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # base slice basis vectors
    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    # Rotate the slice basis around a plane that mixes the x/y slice with the
    # ``w`` axis so that the field varies as the angle changes.
    u = a + b
    v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # origin slice
    origin_pts = sample_slice_image(res_hi, res_hi, o, a, b)
    origin_img = eval_field(origin_pts)
    origin_path = out / "slice_origin.png"
    plt.imsave(origin_path, origin_img)
    paths: Dict[str, str] = {"origin": str(origin_path)}

    # rotated slices
    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        _o, _a, _b = rotate_plane_4d(o, a, b, u, v, angle)
        pts = sample_slice_image(res_hi, res_hi, _o, _a, _b)
        img = eval_field(pts)
        rot_path = out / f"slice_rot_{int(angle):+d}deg.png"
        plt.imsave(rot_path, img)
        paths[f"rot_{angle:+.1f}"] = str(rot_path)

    return {"paths": paths}
    res_hi: int = 128,
    res_coarse: int = 32,
    num_rotated: int = 4,
    z0_steps: int = 1,
    w0_steps: int = 1,
    slopes: np.ndarray | None = None,
    opacity_exp: float = BETA,
    palette: str = "cmy",
    centers: FieldCenters = CENTERS,
    beta: float = BETA,

) -> Dict[str, Any]:
    """Render a small set of slices and return their file paths."""
    """Generate synthetic slices and return their file paths."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = _field_density(res_coarse, centers=centers, beta=beta)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    # High‑resolution origin slice using the requested palette
    rgb, alpha = render_slice(res_hi, res_hi, np.zeros(4, dtype=np.float32), np.eye(4)[0], np.eye(4)[1], centers, np.eye(3), palette)
    origin = composite_rgb_alpha(rgb, alpha)
    origin_path = out_dir / "slice_origin.png"
    plt.imsave(origin_path, origin)

    paths = {"origin": str(origin_path), "coarse_density": str(density_path)}
    return {"paths": paths}


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------
    origin_alpha = _field_density(res_hi, centers=centers, beta=beta)
    origin = np.dstack([origin_alpha] * 3)
    
    """Generate example slices and return their file paths"""
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
    if palette == "cmy":
        rgb = mix_cmy_to_rgb(weights_hi)
    elif palette == "eigen":
        rgb = eigen_palette(weights_hi)
    elif palette == "lineage":
        num_classes = weights_hi.shape[-1]
        palette_rgb = np.zeros((num_classes, 3), dtype=np.float32)
        for i in range(num_classes):
            h, s, v = lineage_hsv_from_address(str(i))
            palette_rgb[i] = hsv_to_rgb([h, s, v])
        weights_norm = weights_hi / (
            np.sum(weights_hi, axis=-1, keepdims=True) + 1e-8
        )
        rgb = weights_norm @ palette_rgb
    else:
        rgb = mix_cmy_to_rgb(weights_hi)
    density_hi = weights_hi.mean(axis=-1)
    alpha = density_to_alpha(density_hi, opacity_exp)
    origin = composite_rgb_alpha(rgb, alpha)
    
    """Generate placeholder slices and basic rendering data.

    Besides writing placeholder images to ``output_dir`` this function now
    computes per-pixel class weights using a randomly initialized class loading
    matrix.  The returned dictionary therefore includes the generated paths as
    well as ``density`` and ``class_weights`` arrays for further processing.    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = np.zeros((res_coarse, res_coarse), dtype=np.float32)
    density_path = out_dir / "coarse_density_map.png"
    plt.imsave(density_path, density, cmap="gray")

    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    a, b = orthonormalize(a, b)

    # Rotation plane (x-z)
    u = a.copy()
    v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    u, v = orthonormalize(u, v)

    centers, V = _demo_centers()

    paths: Dict[str, str] = {}

    img0 = render_slice(res_hi, o, a, b, centers, V)
    origin_path = out_dir / "slice_origin.png"
    plt.imsave(origin_path, img0)
    paths["origin"] = str(origin_path)

    for i in range(num_rotated):
        angle = float(i) * 360.0 / max(num_rotated, 1)
        _o, a_r, b_r = rotate_plane_4d(o, a, b, u, v, angle)
        img = render_slice(res_hi, _o, a_r, b_r, centers, V)
        rot_path = out_dir / f"slice_rot_{int(angle):+d}deg.png"
        plt.imsave(rot_path, img)
        paths[f"rot_{angle:+.1f}"] = str(rot_path)

    paths["coarse_density"] = str(density_path)
    return {"paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine slice renderer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=128)
    parser.add_argument("--res_coarse", type=int, default=32)
    parser.add_argument("--num_rotated", type=int, default=4)
    parser.add_argument("--z0_steps", type=int, default=1)
    parser.add_argument("--w0_steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - manual execution only
    args = _parse_args()
    main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        z0_steps=args.z0_steps,
        w0_steps=args.w0_steps,
    )

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
    class_img = class_weights_to_rgba(class_weights, density, opacity_exp)
    class_path = out_dir / "class_weights_composite.png"
    plt.imsave(class_path, class_img)
    paths["class_weights"] = str(class_path)
# ------------------------------ placeholder main ----------------------------

def main(output_dir: str | Path, **_: Any) -> Dict[str, Any]:
    """Minimal entry point used in tests."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {"paths": {}}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashifine demo renderer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--res_hi", type=int, default=64)
    parser.add_argument("--num_rotated", type=int, default=1)

    parser.add_argument("--opacity_exp", type=float, default=BETA)
    parser.add_argument(
        "--palette",
        type=str,
        default="cmy",
        choices=["cmy", "lineage", "eigen"],
        help="Colour palette for slice rendering ('cmy', 'lineage', or 'eigen')",
    )

    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - manual testing helper
    args = _parse_args()
    main(output_dir=args.output_dir, res_hi=args.res_hi, num_rotated=args.num_rotated)

    main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        opacity_exp=args.opacity_exp,
        palette=args.palette,
    )

    main(output_dir=args.output_dir)
