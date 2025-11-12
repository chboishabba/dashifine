from __future__ import annotations

import pathlib
import sys

import numpy as np

_NEWTEST_DIR = pathlib.Path(__file__).resolve().parents[1] / "newtest"
if str(_NEWTEST_DIR) not in sys.path:
    sys.path.append(str(_NEWTEST_DIR))

import lattice_chsh as LCH


def test_chsh_changes_with_wall_orientation():
    """Tilting the wall frame should modify the reported CHSH value."""

    res = LCH.chsh_on_lattice_frames(N_A=21, N_B=21)
    base_S = res["S"]
    a, ap, b, bp = res["angles"]

    U_A = res["unitaries"]["U_A"]
    U_B = res["unitaries"]["U_B"]
    psi_local = res["states"]["psi_local"]

    theta = 0.37
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=complex,
    )

    psi_tilt = LCH.kron2(U_A, U_B @ rot) @ psi_local
    S_tilt = LCH.chsh_S(psi_tilt, a, ap, b, bp)

    assert not np.isclose(S_tilt, base_S)
