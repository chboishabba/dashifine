import numpy as np
from pathlib import Path
import sys
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashifine.Main_with_rotation import (
    gelu,
    main,
    orthonormalize,
    render,
    p_adic_address_to_hue_saturation,
    rotate_plane_4d,
    rotate_plane,
    class_weights_to_rgba,
)


def test_gelu_is_odd():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    assert np.allclose(gelu(x), -gelu(-x), atol=1e-6)


def test_orthonormalize_returns_unit_orthogonal():
    a = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    ao, bo = orthonormalize(a, b)
    assert np.allclose(np.linalg.norm(ao), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(bo), 1.0, atol=1e-6)
    assert abs(np.dot(ao, bo)) < 1e-6


def test_rotate_plane_4d_produces_orthonormal():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    _, a_rot, b_new = rotate_plane_4d(np.zeros(4, dtype=np.float32), a, b, a, axis, 90.0)
    assert np.allclose(np.linalg.norm(a_rot), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(b_new), 1.0, atol=1e-6)
    assert abs(np.dot(a_rot, b_new)) < 1e-6
    assert np.allclose(a_rot, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_p_adic_palette_maps_address_and_depth():
    addresses = np.array([[0, 1], [2, 3]], dtype=np.int64)
    depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    rgb = render(addresses, depth, palette="p_adic", base=4)

    hue, sat = p_adic_address_to_hue_saturation(addresses, depth, base=4)
    hsv = np.stack([hue, sat, np.ones_like(hue)], axis=-1)
    expected = hsv_to_rgb(hsv)
    assert np.allclose(rgb, expected)




def test_rotate_plane_rotates_basis():
    o = np.zeros(4, dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    _, a_rot, b_new = rotate_plane(o, a, b, axis, 90.0)
    assert np.allclose(a_rot, axis, atol=1e-6)
    assert np.allclose(b_new, b, atol=1e-6)


def test_class_weights_to_rgba_fades_low_density():
    weights = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
    density = np.array([[1.0, 0.0]], dtype=np.float32)
    img = class_weights_to_rgba(weights, density, beta=1.0)
    assert np.allclose(img[0, 0], np.array([0.0, 1.0, 1.0]), atol=1e-6)
    assert np.allclose(img[0, 1], np.ones(3), atol=1e-6)


def test_class_weights_to_rgba_handles_single_channel():
    weights = np.array([[[0.5], [0.0]]], dtype=np.float32)
    density = np.array([[1.0, 0.5]], dtype=np.float32)
    img = class_weights_to_rgba(weights, density, beta=1.0)
    assert img.shape == (1, 2, 3)
    assert np.allclose(img[0, 0], np.array([0.5, 1.0, 1.0]), atol=1e-6)


