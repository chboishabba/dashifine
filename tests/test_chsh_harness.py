import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from newtest.chsh_harness import rotate_in_plane, two_qubit_from_two_local_planes


def test_two_qubit_from_two_local_planes_tracks_inputs():
    uA = np.array([1.0, 0.0], dtype=complex)
    theta = np.pi / 3
    uB = rotate_in_plane(theta) @ uA

    psi = two_qubit_from_two_local_planes(uA, uB)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    expected = (np.kron(
        np.array([1.0, 0.0], dtype=complex),
        np.array([cos_t, sin_t], dtype=complex),
    ) + np.kron(
        np.array([0.0, 1.0], dtype=complex),
        np.array([-sin_t, cos_t], dtype=complex),
    )) / np.sqrt(2)

    assert psi.shape == (4,)
    np.testing.assert_allclose(psi, expected)
