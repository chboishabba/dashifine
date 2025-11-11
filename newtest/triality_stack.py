# ============================================
# triality_stack.py  —  Three-leg SSH "triality" stack
# ============================================
# Dependencies: numpy, matplotlib (only for optional plotting)
# This module DEFINES functions only. It does not execute on import.

from __future__ import annotations
import numpy as np
import numpy.linalg as LA
from math import pi, gcd

# -----------------------------
# Core linear-algebra helpers
# -----------------------------

def R(theta: float) -> np.ndarray:
    """
    Real 2x2 rotation = exp(theta * J) with J = [[0,-1],[1,0]].
    This is your local quarter-turn complex structure in real form.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def eigh_sorted_by_abs(H: np.ndarray):
    """
    Hermitian eigen-decomposition sorted by |eigenvalue|.
    Returns (vals_sorted, vecs_sorted).
    """
    vals, vecs = LA.eigh(H)
    order = np.argsort(np.abs(vals))
    return vals[order], vecs[:, order]

# -----------------------------
# Single-leg (open) chiral SSH
# -----------------------------

def build_T_open(
    N: int,
    t1: float,
    t2: float,
    thetas_intra: np.ndarray,
    thetas_inter: np.ndarray,
    domain_wall_at: int,
    swap_strengths_at_wall: bool = True,
) -> np.ndarray:
    """
    Build the chiral SSH 'T' block for one OPEN leg:
      - A_i <- B_i with weight t1 * R(theta_intra[i])
      - A_{i+1} <- B_i with weight t2 * R(theta_inter[i]) for i=0..N-2
      - If swap_strengths_at_wall, swap (t1 <-> t2) once at cell 'domain_wall_at'.
    Shapes:
      A-dimension = 2*N   (2 internal dims per site)
      B-dimension = 2*N
    """
    assert thetas_intra.shape == (N,)
    assert thetas_inter.shape == (N-1,)
    A_dim = 2 * N
    B_dim = 2 * N
    T = np.zeros((A_dim, B_dim), dtype=float)

    for i in range(N):
        t1i, t2i = t1, t2
        if swap_strengths_at_wall and i == domain_wall_at:
            t1i, t2i = t2, t1

        # Intracell: A_i <- B_i
        a = slice(2 * i, 2 * i + 2)
        b = slice(2 * i, 2 * i + 2)
        T[a, b] += t1i * R(thetas_intra[i])

        # Intercell: A_{i+1} <- B_i
        if i + 1 < N:
            ap1 = slice(2 * (i + 1), 2 * (i + 1) + 2)
            T[ap1, b] += t2i * R(thetas_inter[i])

    return T

def build_chiral_H_from_T(T: np.ndarray) -> np.ndarray:
    """
    Build the chiral Hamiltonian for a single leg:
      H = [[0, T],
           [T^T, 0]]
    """
    A_dim, B_dim = T.shape
    Z_A = np.zeros((A_dim, A_dim), dtype=float)
    Z_B = np.zeros((B_dim, B_dim), dtype=float)
    return np.block([[Z_A, T],
                     [T.T, Z_B]])

def gamma_operator_single_leg(N: int) -> np.ndarray:
    """
    Gamma = diag(+I_A, -I_B) for a single leg with 2D internal per site.
    """
    I_A = np.eye(2 * N, dtype=float)
    I_B = np.eye(2 * N, dtype=float)
    Z   = np.zeros_like(I_A)
    return np.block([[ I_A, Z],
                     [ Z, -I_B]])

# -----------------------------
# Three-leg stack (triality)
# -----------------------------

def build_triality_stack_H(
    N: int,
    t1: float,
    t2: float,
    phases_leg: list[tuple[np.ndarray, np.ndarray]],
    domain_wall_at: int,
    g_perp: float,
    interleg_phase_shifts: tuple[float, float, float] = (0.0, 2 * pi / 3, 4 * pi / 3),
) -> np.ndarray:
    """
    Build the full Hamiltonian for a three-leg (triality) stack:
      - Each leg is an open, chiral SSH with one domain wall and gauge-covariant links.
      - Inter-leg coupling g_perp couples A<->A and B<->B at the same cell index,
        using diagonal 2x2 rotation blocks R(phi) with 120° shifts across legs.

    phases_leg: list of 3 items, each is (thetas_intra, thetas_inter) for that leg.
      thetas_intra.shape = (N,), thetas_inter.shape = (N-1,)
    """
    L = 3
    assert len(phases_leg) == L
    leg_dim = 4 * N  # (A:2N + B:2N)
    H = np.zeros((L * leg_dim, L * leg_dim), dtype=float)

    # Diagonal leg Hamiltonians
    for ell in range(L):
        thetas_intra, thetas_inter = phases_leg[ell]
        T = build_T_open(
            N, t1, t2, thetas_intra, thetas_inter,
            domain_wall_at=domain_wall_at,
            swap_strengths_at_wall=True
        )
        H_leg = build_chiral_H_from_T(T)
        s = ell * leg_dim
        H[s:s + leg_dim, s:s + leg_dim] = H_leg

    # Inter-leg couplers with 120° phases
    shifts = interleg_phase_shifts
    for ell in range(L):
        nxt = (ell + 1) % L
        phi = shifts[ell % len(shifts)]
        Rphi = R(phi)
        # A-block coupling (2N x 2N)
        A_ell = slice(ell * leg_dim, ell * leg_dim + 2 * N)
        A_nxt = slice(nxt * leg_dim, nxt * leg_dim + 2 * N)
        Ablock = np.kron(np.eye(N), Rphi)
        H[A_ell, A_nxt] += g_perp * Ablock
        H[A_nxt, A_ell] += g_perp * Ablock.T
        # B-block coupling (2N x 2N)
        B_ell = slice(ell * leg_dim + 2 * N, ell * leg_dim + 4 * N)
        B_nxt = slice(nxt * leg_dim + 2 * N, nxt * leg_dim + 4 * N)
        Bblock = np.kron(np.eye(N), Rphi)
        H[B_ell, B_nxt] += g_perp * Bblock
        H[B_nxt, B_ell] += g_perp * Bblock.T

    return H

# -----------------------------
# Convenience builders & checks
# -----------------------------

def intercell_residues_bump(N: int, wall: int, width: int = 4) -> np.ndarray:
    """
    Make a small mod-6 residue "bump" centered at 'wall' for intercell phases.
    Returns int residues r(i) in {0..5} of shape (N-1,).
    """
    r = np.zeros(N - 1, dtype=int)
    for j in range(-width, width + 1):
        idx = wall + j
        if 0 <= idx < N - 1:
            r[idx] = (j % 6)
    return r

def make_leg_phases_with_offset(N: int, wall: int, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Intra phases = 0; inter phases = (2π/6)*bump + offset.
    """
    thetas_intra = np.zeros(N, dtype=float)
    r = intercell_residues_bump(N, wall, width=4)
    thetas_inter = (2 * pi * r / 6.0).astype(float) + offset
    return thetas_intra, thetas_inter

def gamma_triality_stack(N: int) -> np.ndarray:
    """
    Gamma for the full three-leg stack = block-diag(Gamma_single_leg, Gamma_single_leg, Gamma_single_leg).
    """
    G1 = gamma_operator_single_leg(N)
    Z  = np.zeros_like(G1)
    return np.block([
        [ G1, Z,  Z ],
        [ Z,  G1, Z ],
        [ Z,  Z,  G1],
    ])

# NOTE: Add your own plotting or scanning functions in your runner script.
# This module intentionally does NOT execute anything on import.


def intercell_residues_bump_width(N: int, wall: int, width: int = 4) -> np.ndarray:
    """
    Integer bump around 'wall' (symmetric), shape (N-1,). Values are integers (can be negative),
    which we later map to angles via modulus M.
    """
    r = np.zeros(N-1, dtype=int)
    for j in range(-width, width+1):
        idx = wall + j
        if 0 <= idx < N-1:
            r[idx] = j  # signed bump
    return r

def make_leg_phases_with_modulus(N: int, wall: int, M: int, offset: float = 0.0, width: int = 4):
    """
    Intra phases = 0; inter phases quantized by modulus M:
        theta_inter[i] = offset + 2π * (r[i] mod M) / M
    If you prefer signed mapping, replace (r % M) by np.mod(r+M, M).
    """
    thetas_intra = np.zeros(N, dtype=float)
    r = intercell_residues_bump_width(N, wall, width=width)
    # Map integer residues into [0..M-1], then to angles
    r_mod = np.mod(r, M)
    thetas_inter = (2*np.pi * r_mod / float(M)).astype(float) + float(offset)
    return thetas_intra, thetas_inter

def quantize_phases(theta: np.ndarray, k: int) -> np.ndarray:
    """
    Round each phase to nearest multiple of 2π/k.
    """
    step = 2*np.pi/float(k)
    return np.round(theta / step) * step


def quantize_phases_randomized(theta: np.ndarray, k: int, p_round: float = 1.0, rng=None) -> np.ndarray:
    """
    With prob p_round, round theta to nearest multiple of 2π/k; else leave as-is.
    """
    if rng is None:
        rng = np.random.default_rng()
    step = 2*np.pi/float(k)
    out = theta.copy()
    mask = rng.random(theta.shape) < p_round
    out[mask] = np.round(theta[mask] / step) * step
    return out

def build_single_leg_open_modulus_quantized_random(
    N: int, t1: float, t2: float, wall: int, M: int, k_quant: int,
    p_round: float = 1.0, width: int = 4, rng=None
):
    """
    As build_single_leg_open_modulus_quantized, but randomizes phase rounding
    so repeated runs with different RNG seeds explore granular noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    thetas_intra, thetas_inter = make_leg_phases_with_modulus(N, wall, M, width=width)
    thetas_inter_q = quantize_phases_randomized(thetas_inter, k_quant, p_round=p_round, rng=rng)
    T = build_T_open(N, t1, t2, thetas_intra, thetas_inter_q, wall, swap_strengths_at_wall=True)
    H = build_chiral_H_from_T(T)
    vals, vecs = eigh_sorted_by_abs(H)
    return H, vals, vecs


def add_gaussian_phase_noise(theta: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise N(0, sigma^2) to each phase entry.
    """
    if rng is None:
        rng = np.random.default_rng()
    return theta + rng.normal(loc=0.0, scale=float(sigma), size=theta.shape)

def build_single_leg_open_modulus_gaussian(
    N: int, t1: float, t2: float, wall: int, M: int, sigma: float,
    width: int = 4, rng=None
):
    """
    Like build_single_leg_open_modulus, but add Gaussian phase noise (sigma) to inter-cell phases.
    """
    if rng is None:
        rng = np.random.default_rng()
    thetas_intra, thetas_inter = make_leg_phases_with_modulus(N, wall, M, width=width)
    thetas_inter_n = add_gaussian_phase_noise(thetas_inter, sigma=sigma, rng=rng)
    T = build_T_open(N, t1, t2, thetas_intra, thetas_inter_n, wall, swap_strengths_at_wall=True)
    H = build_chiral_H_from_T(T)
    vals, vecs = eigh_sorted_by_abs(H)
    return H, vals, vecs


def build_single_leg_open_modulus_quantized_then_gaussian(
    N: int, t1: float, t2: float, wall: int, M: int,
    k_quant: int, p_round: float, sigma: float,
    width: int = 4, rng=None
):
    """
    Inter-cell phases for one leg:
      1) start from modular pattern (M)
      2) randomized rounding to 2π/k_quant with prob p_round
      3) add i.i.d. Gaussian jitter (sigma radians)
    Returns (H, vals_sorted, vecs_sorted).
    """
    if rng is None:
        rng = np.random.default_rng()
    thetas_intra, thetas_inter = make_leg_phases_with_modulus(N, wall, M, width=width)
    thetas_q  = quantize_phases_randomized(thetas_inter, k_quant, p_round=p_round, rng=rng)
    thetas_qg = add_gaussian_phase_noise(thetas_q, sigma=sigma, rng=rng)
    T = build_T_open(N, t1, t2, thetas_intra, thetas_qg, wall, swap_strengths_at_wall=True)
    H = build_chiral_H_from_T(T)
    vals, vecs = eigh_sorted_by_abs(H)
    return H, vals, vecs


def lcm(a: int, b: int) -> int:
    return abs(a*b) // gcd(a, b) if a and b else 0

def lcm_list(Ms):
    out = 1
    for m in Ms:
        out = lcm(out, int(m))
    return out

def wrap_angle(theta: np.ndarray) -> np.ndarray:
    # Wrap to principal value (-pi, pi]
    return np.angle(np.exp(1j*theta))

def make_leg_phases_with_composite_moduli(
    N: int,
    wall: int,
    Ms: list[int],
    *,
    offset: float = 0.0,
    width: int = 4,
    mode: str = "lcm",   # "lcm" or "sum"
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build inter-cell phases from a *set* of moduli Ms.
    - mode="lcm":   effective modulus M_eff = lcm(Ms); theta = 2π * (r mod M_eff) / M_eff
      -> Gives harmonics at divisors and clean CRT-like composite.
    - mode="sum":   theta = Σ_i w_i * [2π * (r mod M_i) / M_i], then wrapped to (-π, π]
      -> Produces visible multi-harmonic 'beats'; default equal weights.

    Returns (thetas_intra, thetas_inter) with shape (N,), (N-1,).
    """
    thetas_intra = np.zeros(N, dtype=float)
    r = intercell_residues_bump_width(N, wall, width=width)  # signed integers

    if mode == "lcm":
        M_eff = lcm_list(Ms)
        r_mod = np.mod(r, M_eff)
        thetas_inter = (2*np.pi * r_mod / float(M_eff)).astype(float) + float(offset)
    elif mode == "sum":
        Ms = [int(m) for m in Ms]
        if weights is None:
            weights = np.ones(len(Ms), dtype=float) / float(len(Ms))
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / np.sum(weights)
        theta_sum = np.zeros_like(r, dtype=float)
        for w, Mi in zip(weights, Ms):
            theta_sum += w * (2*np.pi * np.mod(r, Mi) / float(Mi))
        thetas_inter = wrap_angle(theta_sum + float(offset))
    else:
        raise ValueError("mode must be 'lcm' or 'sum'")

    return thetas_intra, thetas_inter

def build_single_leg_open_composite_quantized_then_gaussian(
    N: int, t1: float, t2: float, wall: int,
    Ms: list[int],
    *, k_quant: int, p_round: float, sigma: float,
    width: int = 4,
    mode: str = "lcm",
    weights: np.ndarray | None = None,
    rng=None
):
    """
    Composite-moduli pipeline for one leg:
      phases := composite(Ms, mode)  -> randomized rounding to 2π/k_quant (p_round)
              -> add Gaussian jitter N(0, sigma^2)
    """
    if rng is None:
        rng = np.random.default_rng()

    thetas_intra, thetas_inter = make_leg_phases_with_composite_moduli(
        N, wall, Ms, width=width, mode=mode, weights=weights
    )
    thetas_q  = quantize_phases_randomized(thetas_inter, k_quant, p_round=p_round, rng=rng)
    thetas_qg = add_gaussian_phase_noise(thetas_q, sigma=sigma, rng=rng)

    T = build_T_open(N, t1, t2, thetas_intra, thetas_qg, wall, swap_strengths_at_wall=True)
    H = build_chiral_H_from_T(T)
    vals, vecs = eigh_sorted_by_abs(H)
    return H, vals, vecs
