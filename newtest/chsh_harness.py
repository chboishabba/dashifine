# ============================================
# chsh_harness.py  —  Bell/CHSH evaluator
# ============================================
# Dependencies: numpy (only)
# This module DEFINES functions only. It does not execute on import.
#
# Part A: Pure 2-qubit CHSH evaluator (angles on a "J-plane").
# Part B: Optional STUB showing how to map lattice-local 2D planes (cos,sin at wall cells)
#         to those qubit measurement axes. (You can tailor this to your extraction.)

from __future__ import annotations
import numpy as np
import numpy.linalg as LA

# -----------------------------
# Part A: 2-qubit CHSH evaluator
# -----------------------------

def pauli_matrices():
    """Standard complex Pauli matrices (for compact CHSH math)."""
    sx = np.array([[0, 1],
                   [1, 0]], dtype=complex)
    sy = np.array([[0, -1j],
                   [1j, 0]], dtype=complex)
    sz = np.array([[1, 0],
                   [0, -1]], dtype=complex)
    return sx, sy, sz

def meas_op(angle: float) -> np.ndarray:
    """
    Single-qubit measurement operator along angle 'angle' in the x–z plane:
       M(angle) = cos(angle)*σ_z + sin(angle)*σ_x
    This matches rotating within the local (cos,sin) plane where your J acts like "i".
    """
    sx, _, sz = pauli_matrices()
    return np.cos(angle) * sz + np.sin(angle) * sx

def kron2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)

def bell_state_phi_plus() -> np.ndarray:
    """
    |Φ+> = (|00> + |11>)/√2
    Good maximally entangled test state.
    """
    v = np.zeros((4,), dtype=complex)
    v[0] = 1/np.sqrt(2)
    v[3] = 1/np.sqrt(2)
    return v

def bell_state_psi_plus() -> np.ndarray:
    """
    |Ψ+> = (|01> + |10>)/√2
    Another useful Bell state.
    """
    v = np.zeros((4,), dtype=complex)
    v[1] = 1/np.sqrt(2)
    v[2] = 1/np.sqrt(2)
    return v

def chsh_S(psi: np.ndarray, a: float, a_p: float, b: float, b_p: float) -> float:
    """
    Compute the CHSH parameter:
      S = E(a,b) + E(a,b') + E(a',b) - E(a',b')
    where E(α,β) = <psi| M_A(α) ⊗ M_B(β) |psi>.
    """
    MA_a   = meas_op(a)
    MA_ap  = meas_op(a_p)
    MB_b   = meas_op(b)
    MB_bp  = meas_op(b_p)

    E_ab   = np.vdot(psi, kron2(MA_a,  MB_b ) @ psi).real
    E_abp  = np.vdot(psi, kron2(MA_a,  MB_bp) @ psi).real
    E_apb  = np.vdot(psi, kron2(MA_ap, MB_b ) @ psi).real
    E_apbp = np.vdot(psi, kron2(MA_ap, MB_bp) @ psi).real

    S = E_ab + E_abp + E_apb - E_apbp
    return float(S)

def tsirelson_angles():
    """
    A standard choice achieving S = 2√2 on |Φ+>:
      a = 0, a' = π/2, b = π/4, b' = -π/4
    """
    a   = 0.0
    ap  = 0.5 * np.pi
    b   = 0.25 * np.pi
    bp  = -0.25 * np.pi
    return a, ap, b, bp

# -----------------------------
# Part B: Mapping lattice -> qubit (STUB)
# -----------------------------

def extract_local_plane_basis_at_wall(
    full_vec: np.ndarray,
    N: int,
    wall_cell: int,
    which_block: str = "A",
) -> np.ndarray:
    """
    STUB: Given a normalized eigenvector 'full_vec' from your lattice model,
    extract the 2D internal (cos,sin) vector at the wall cell for a chosen sublattice block ("A" or "B"),
    and return a normalized 2-vector (complex for convenience).
    - full_vec layout (single leg): [A(0..2N-1), B(0..2N-1)], each A/B has 2 entries per cell.
    NOTE: For a three-leg stack, pass a segment for the leg you care about.
    """
    assert which_block in ("A", "B")
    if which_block == "A":
        off = 0
    else:
        off = 2 * N
    start = off + 2 * wall_cell
    v2 = full_vec[start:start + 2].astype(complex)  # treat as complex 2-vector
    nrm = np.linalg.norm(v2)
    if nrm > 0:
        v2 = v2 / nrm
    return v2

def projector_on_direction(vec2: np.ndarray) -> np.ndarray:
    """
    Rank-1 projector |v><v| on a 2D complex vector (normalized).
    """
    v = vec2.reshape(2, 1)
    return v @ v.conj().T

def rotate_in_plane(angle: float) -> np.ndarray:
    """
    Complex 2x2 rotation for convenience (matches real-plane rotation + complex phase).
    Equivalent to exp(angle * J) if you embed J as [[0,-1],[1,0]] and then lift to C.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s],
                     [s,  c]], dtype=complex)

def two_qubit_from_two_local_planes(uA: np.ndarray, uB: np.ndarray) -> np.ndarray:
    """
    Build a simple maximally entangled state from two local 2D unit vectors
    (use their local 'z' as |0>, orthogonal as |1>), return |Φ+> in that local basis.
    This is an abstract 2-qubit state (4-vector).
    """
    # Local orthonormal frames on each side: {u, u_perp}
    # Build u_perp by a +90° rotation
    Jc = np.array([[0, -1],
                   [1,  0]], dtype=complex)
    def ortho(v):
        w = (Jc @ v.reshape(2,1)).ravel()
        # Re-orthonormalize relative to v
        w = w - (np.vdot(v, w) * v)
        n = np.linalg.norm(w)
        return w / n if n > 0 else w

    uA = uA / (np.linalg.norm(uA) + 1e-15)
    uB = uB / (np.linalg.norm(uB) + 1e-15)
    a0, a1 = uA, ortho(uA)
    b0, b1 = uB, ortho(uB)

    # Map {|0>_A, |1>_A}⊗{|0>_B, |1>_B} to C^4 canonical basis
    # Construct |Φ+> = (|00> + |11>)/√2 in that local basis
    psi = (np.kron(a0, b0) + np.kron(a1, b1)) / np.sqrt(2)
    return psi.reshape(4)

# NOTE:
# - The CHSH evaluator above (Part A) operates entirely in a 2-qubit space.
# - To make it "touch" the lattice:
#     1) Compute your lattice eigenvectors for two separated wall modes (systems A and B).
#     2) Use 'extract_local_plane_basis_at_wall' on each to get two local 2D directions.
#     3) Build a 2-qubit Bell state in those local frames via 'two_qubit_from_two_local_planes'.
#     4) Evaluate S with 'chsh_S' at Tsirelson angles (or custom).
#
# This keeps the numerics light and cleanly separates: lattice → local measurement frames → CHSH.
#
# This module intentionally does NOT execute anything on import.
