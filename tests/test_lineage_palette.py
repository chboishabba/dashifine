import numpy as np
from pathlib import Path
import sys
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PATCH_DROPIN_SUGGESTED import (
    lineage_hue_from_address,
    render_slice,
    sample_slice_points,
    field_and_classes,
)


def test_lineage_hue_from_address_parses_base_p():
    h, s, v = lineage_hue_from_address("123.5", base=4)
    assert abs(h - 0.890625) < 1e-6
    assert abs(s - 0.5) < 1e-6
    assert abs(v - 0.75) < 1e-6


def test_render_slice_lineage_palette_matches_hsv_mapping():
    H = W = 1
    origin = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    centers = [{"mu": origin, "sigma": np.ones(4, dtype=np.float32), "w": 1.0}]
    V = np.eye(1, dtype=np.float32)

    rgb = render_slice(
        H,
        W,
        origin,
        a,
        b,
        centers,
        V,
        palette="lineage",
        bg=np.zeros(3, dtype=np.float32),
        beta=0.0,
    )

    pts = sample_slice_points(H, W, origin, a, b)
    rho, Wc = field_and_classes(pts, centers, V)
    top_idx = np.argmax(Wc, axis=1)
    depth = np.max(Wc, axis=1)
    hsv = np.zeros((Wc.shape[0], 3), dtype=np.float32)
    for i, (idx, d) in enumerate(zip(top_idx, depth)):
        d_clip = np.clip(d, 0.0, 0.999)
        addr = f"{int(idx)}.{int(d_clip * 1000):03d}"
        h, s, v = lineage_hue_from_address(addr)
        hsv[i] = [h, s, v]
    expected_rgb = hsv_to_rgb(hsv).reshape(H, W, 3)
    assert np.allclose(rgb, expected_rgb)


def test_render_slice_lineage_palette_uses_centre_addresses():
    H, W = 1, 2
    origin = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    sigma = np.full(4, 0.1, dtype=np.float32)
    centers = [
        {"mu": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32), "sigma": sigma, "w": 1.0, "addr": "0.000"},
        {"mu": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "sigma": sigma, "w": 1.0, "addr": "1.000"},
    ]
    V = np.eye(len(centers), dtype=np.float32)

    img = render_slice(
        H,
        W,
        origin,
        a,
        b,
        centers,
        V,
        palette="lineage",
        bg=np.zeros(3, dtype=np.float32),
        beta=0.0,
    )

    pts = sample_slice_points(H, W, origin, a, b)
    _, Wc = field_and_classes(pts, centers, V)
    top_idx = np.argmax(Wc, axis=1)
    centre_hsv = [lineage_hue_from_address(c["addr"]) for c in centers]
    centre_rgb = hsv_to_rgb(np.array(centre_hsv, dtype=np.float32))
    expected = centre_rgb[top_idx].reshape(H, W, 3)

    assert img.shape == (H, W, 3)
    assert np.allclose(img, expected)
