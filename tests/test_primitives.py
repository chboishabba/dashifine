import numpy as np
from Main_with_rotation import gelu, orthonormalize, rotate_plane


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


def test_rotate_plane_produces_orthonormal():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    _, a_rot, b_new = rotate_plane(np.zeros(4, dtype=np.float32), a, b, axis, 90.0)
    assert np.allclose(np.linalg.norm(a_rot), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(b_new), 1.0, atol=1e-6)
    assert abs(np.dot(a_rot, b_new)) < 1e-6
    assert np.allclose(a_rot, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
