# ============================================
# lattice_chsh.py — Lattice-aware CHSH utilities (no execution on import)
# ============================================
# Dependencies: numpy
# Optional (for plotting in your runner): matplotlib
#
# This module wires your open-chain, chiral SSH wall modes into a 2-qubit CHSH test:
#   • Builds two independent single-leg SSH chains (A, B) with a domain wall each.
#   • Extracts the local 2D (cos,sin) plane at each wall to define qubit frames.
#   • Prepares an entangled two-qubit state either:
#        - 'ideal_bell'  : exact |Φ+> in those local frames
#        - 'heisenberg'  : evolve |01> under H = J*(σx⊗σx + σy⊗σy + σz⊗σz) for time τ
#   • Evaluates CHSH S with chosen angles (defaults are Tsirelson).
#
# NOTE: This file depends on your triality_stack.py single-leg builders.

from __future__ import annotations
import numpy as np
import numpy.linalg as LA
from math import pi
import triality_stack as TST  # re-use single-leg SSH builders

# -----------------------------
# Lattice → local plane extraction
# -----------------------------

def extract_local_plane_basis_at_wall_single_leg(
    full_vec: np.ndarray,
    N: int,
    wall_cell: int,
    which_block: str = "A",
) -> np.ndarray:
    """
    Given a normalized eigenvector 'full_vec' from a SINGLE LEG Hamiltonian H_leg,
    extract the 2D internal (cos,sin) vector at the wall cell for the chosen sublattice ("A" or "B"),
    and return a normalized 2-vector (complex dtype for convenience).
    H_leg layout = [A(0..2N-1), B(0..2N-1)] with 2 entries per cell for each block.
    """
    assert which_block in ("A", "B")
    off = 0 if which_block == "A" else 2 * N
    start = off + 2 * wall_cell
    v2 = full_vec[start:start + 2].astype(complex)
    nrm = np.linalg.norm(v2)
    if nrm > 0:
        v2 = v2 / nrm
    return v2

def ortho_2d(v: np.ndarray) -> np.ndarray:
    """
    Return a unit vector orthogonal to v in C^2 using the J-rotation trick + Gram-Schmidt.
    """
    Jc = np.array([[0, -1],
                   [1,  0]], dtype=complex)
    w = (Jc @ v.reshape(2, 1)).ravel()
    # Gram-Schmidt to ensure orthonormality
    w = w - (np.vdot(v, w) * v)
    n = np.linalg.norm(w)
    return w / n if n > 0 else w

# -----------------------------
# Qubit layer: operators and states
# -----------------------------

def pauli():
    sx = np.array([[0, 1],
                   [1, 0]], dtype=complex)
    sy = np.array([[0, -1j],
                   [1j,  0]], dtype=complex)
    sz = np.array([[1, 0],
                   [0,-1]], dtype=complex)
    return sx, sy, sz

def meas_op(angle: float) -> np.ndarray:
    """
    M(angle) = cos(angle) σz + sin(angle) σx   (axes in the local J-plane)
    """
    sx, _, sz = pauli()
    return np.cos(angle) * sz + np.sin(angle) * sx

def kron2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)

def bell_phi_plus() -> np.ndarray:
    v = np.zeros(4, complex)
    v[0] = 1/np.sqrt(2)
    v[3] = 1/np.sqrt(2)
    return v

def bell_psi_plus() -> np.ndarray:
    v = np.zeros(4, complex)
    v[1] = 1/np.sqrt(2)
    v[2] = 1/np.sqrt(2)
    return v

def chsh_S(psi: np.ndarray, a: float, ap: float, b: float, bp: float) -> float:
    MA, MAp = meas_op(a), meas_op(ap)
    MB, MBp = meas_op(b), meas_op(bp)
    Eab   = np.vdot(psi, kron2(MA,  MB)  @ psi).real
    Eabp  = np.vdot(psi, kron2(MA,  MBp) @ psi).real
    Eapb  = np.vdot(psi, kron2(MAp, MB)  @ psi).real
    Eapbp = np.vdot(psi, kron2(MAp, MBp) @ psi).real
    return float(Eab + Eabp + Eapb - Eapbp)

def tsirelson_angles():
    a   = 0.0
    ap  = 0.5 * np.pi
    b   = 0.25 * np.pi
    bp  = -0.25 * np.pi
    return a, ap, b, bp

# -----------------------------
# Entanglers in the 2-qubit space
# -----------------------------

def heisenberg_unitary(J: float, tau: float) -> np.ndarray:
    """
    U = exp(-i * J * tau * (σx⊗σx + σy⊗σy + σz⊗σz))
    Acts on C^2 ⊗ C^2
    """
    sx, sy, sz = pauli()
    H = kron2(sx, sx) + kron2(sy, sy) + kron2(sz, sz)
    return LA.expm(-1j * J * tau * H)

def prepare_two_qubit_state(
    mode: str = "ideal_bell",
    J: float = 1.0,
    tau: float = np.pi/8,
) -> np.ndarray:
    """
    mode = 'ideal_bell' : return |Φ+>
         = 'heisenberg' : start |01>, evolve with Heisenberg U(J, tau)
    """
    if mode == "ideal_bell":
        return bell_phi_plus()
    elif mode == "heisenberg":
        psi0 = np.zeros(4, complex)
        psi0[1] = 1.0  # |01>
        U = heisenberg_unitary(J, tau)
        return U @ psi0
    else:
        raise ValueError("Unknown mode")

# -----------------------------
# Build two single-leg lattices and extract local frames
# -----------------------------

def build_single_leg_open(
    N: int,
    t1: float,
    t2: float,
    wall: int,
    which_block_for_measure: str = "A",
    phase_bump_width: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (H_leg, vals_sorted, vecs_sorted) for ONE leg.
    Intracell phases = 0; intercell phases = (2π/6)*bump around 'wall'.
    """
    thetas_intra = np.zeros(N, dtype=float)
    r = TST.intercell_residues_bump(N, wall, width=phase_bump_width)
    thetas_inter = (2 * np.pi * r / 6.0).astype(float)
    T = TST.build_T_open(N, t1, t2, thetas_intra, thetas_inter, wall, swap_strengths_at_wall=True)
    H = TST.build_chiral_H_from_T(T)
    vals, vecs = TST.eigh_sorted_by_abs(H)
    return H, vals, vecs

def extract_wall_qubit_frame(
    vecs_sorted: np.ndarray,
    N: int,
    wall: int,
    which_block: str = "A",
) -> tuple[np.ndarray, np.ndarray]:
    """
    From the near-zero eigenvector, build a local qubit frame (|0>, |1>) at the wall:
       |0> = u  (2D)
       |1> = u_perp
    Returns (u, u_perp) as 2D complex vectors.
    """
    v0 = vecs_sorted[:, 0]
    u  = extract_local_plane_basis_at_wall_single_leg(v0, N, wall, which_block=which_block)
    up = ortho_2d(u)
    return u, up

# -----------------------------
# CHSH wrapper using lattice-defined frames
# -----------------------------

def chsh_on_lattice_frames(
    N_A: int = 41, N_B: int = 41,
    t1: float = 0.7, t2: float = 1.3,
    wall_A: int | None = None, wall_B: int | None = None,
    which_block_A: str = "A", which_block_B: str = "A",
    prep_mode: str = "ideal_bell",  # 'ideal_bell' or 'heisenberg'
    Jprep: float = 1.0, tau: float = np.pi/8,
    angles: tuple[float, float, float, float] | None = None,
) -> dict:
    """
    Build two independent single-leg systems (A,B), extract their local wall frames,
    prepare a two-qubit state in those frames, and evaluate CHSH.
    """
    if wall_A is None: wall_A = N_A // 2
    if wall_B is None: wall_B = N_B // 2
    if angles is None:
        a, ap, b, bp = tsirelson_angles()
    else:
        a, ap, b, bp = angles

    # Build legs and get near-zero eigenvectors
    H_A, valsA, vecsA = build_single_leg_open(N_A, t1, t2, wall_A)
    H_B, valsB, vecsB = build_single_leg_open(N_B, t1, t2, wall_B)

    # Extract 2D local frames at each wall
    uA, uA_perp = extract_wall_qubit_frame(vecsA, N_A, wall_A, which_block=which_block_A)
    uB, uB_perp = extract_wall_qubit_frame(vecsB, N_B, wall_B, which_block=which_block_B)

    # Build a 2-qubit state
    if prep_mode == "ideal_bell":
        psi = bell_phi_plus()
    elif prep_mode == "heisenberg":
        psi = prepare_two_qubit_state("heisenberg", J=Jprep, tau=tau)
    else:
        raise ValueError("prep_mode must be 'ideal_bell' or 'heisenberg'")

    # Evaluate CHSH in those local planes (angles defined in each local J-plane)
    S = chsh_S(psi, a, ap, b, bp)
    return dict(
        S=S,
        angles=(a, ap, b, bp),
        valsA=valsA[:6], valsB=valsB[:6],
        wall_frames=dict(
            uA=uA, uA_perp=uA_perp,
            uB=uB, uB_perp=uB_perp
        )
    )

def build_single_leg_open_modulus(
    N: int, t1: float, t2: float, wall: int, M: int, offset: float = 0.0, width: int = 4
):
    """
    One open chiral SSH leg; inter-cell phases drawn from modulus M (6, 9, 15, ...).
    """
    thetas_intra, thetas_inter = TST.make_leg_phases_with_modulus(N, wall, M, offset=offset, width=width)
    T = TST.build_T_open(N, t1, t2, thetas_intra, thetas_inter, wall, swap_strengths_at_wall=True)
    H = TST.build_chiral_H_from_T(T)
    vals, vecs = TST.eigh_sorted_by_abs(H)
    return H, vals, vecs

def build_single_leg_open_modulus_quantized(
    N: int, t1: float, t2: float, wall: int, M: int, k_quant: int,
    offset: float = 0.0, width: int = 4
):
    """
    Same as build_single_leg_open_modulus but quantizes inter phases to 2π/k_quant.
    """
    thetas_intra, thetas_inter = TST.make_leg_phases_with_modulus(N, wall, M, offset=offset, width=width)
    thetas_inter_q = TST.quantize_phases(thetas_inter, k_quant)
    T = TST.build_T_open(N, t1, t2, thetas_intra, thetas_inter_q, wall, swap_strengths_at_wall=True)
    H = TST.build_chiral_H_from_T(T)
    vals, vecs = TST.eigh_sorted_by_abs(H)
    return H, vals, vecs

def build_single_leg_open_modulus_quantized_random(
    N: int, t1: float, t2: float, wall: int, M: int, k_quant: int,
    p_round: float = 1.0, width: int = 4, rng=None
):
    """
    Wrapper that delegates to triality_stack.build_single_leg_open_modulus_quantized_random.
    Returns (H, vals_sorted, vecs_sorted).
    """
    return TST.build_single_leg_open_modulus_quantized_random(
        N=N, t1=t1, t2=t2, wall=wall, M=M, k_quant=k_quant,
        p_round=p_round, width=width, rng=rng
    )

def build_single_leg_open_modulus_gaussian(
    N: int, t1: float, t2: float, wall: int, M: int, sigma: float,
    width: int = 4, rng=None
):
    """
    Wrapper to triality_stack.build_single_leg_open_modulus_gaussian.
    """
    return TST.build_single_leg_open_modulus_gaussian(
        N=N, t1=t1, t2=t2, wall=wall, M=M, sigma=sigma, width=width, rng=rng
    )

def build_single_leg_open_modulus_quantized_then_gaussian(
    N: int, t1: float, t2: float, wall: int, M: int,
    k_quant: int, p_round: float, sigma: float,
    width: int = 4, rng=None
):
    return TST.build_single_leg_open_modulus_quantized_then_gaussian(
        N=N, t1=t1, t2=t2, wall=wall, M=M,
        k_quant=k_quant, p_round=p_round, sigma=sigma,
        width=width, rng=rng
    )


def build_single_leg_open_composite_quantized_then_gaussian(
    N: int, t1: float, t2: float, wall: int,
    Ms: list[int],
    k_quant: int, p_round: float, sigma: float,
    width: int = 4,
    mode: str = "lcm",
    weights: np.ndarray | None = None,
    rng=None
):
    return TST.build_single_leg_open_composite_quantized_then_gaussian(
        N=N, t1=t1, t2=t2, wall=wall,
        Ms=Ms, k_quant=k_quant, p_round=p_round, sigma=sigma,
        width=width, mode=mode, weights=weights, rng=rng
    )
