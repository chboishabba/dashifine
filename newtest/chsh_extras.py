# ============================================
# chsh_extras.py — CHSH scans, no-signalling, and simple noise (no auto-exec)
# ============================================
from __future__ import annotations
import numpy as np
import numpy.linalg as LA
from scipy.linalg import expm


# ---------- Core qubit ops ----------
def pauli():
    sx = np.array([[0,1],[1,0]], complex)
    sy = np.array([[0,-1j],[1j,0]], complex)
    sz = np.array([[1,0],[0,-1]], complex)
    return sx, sy, sz

def meas_op(angle: float) -> np.ndarray:
    sx, _, sz = pauli()
    return np.cos(angle)*sz + np.sin(angle)*sx

def kron2(A,B): return np.kron(A,B)

# ---------- State helpers ----------
def ket_to_rho(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape(-1,1)
    return psi @ psi.conj().T

def depolarize_rho(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Two-qubit depolarizing channel:
      ρ -> (1-p) ρ + p * I/4
    """
    I4 = np.eye(4, dtype=complex)
    return (1.0 - p) * rho + p * I4 / 4.0

# ---------- Expectation / CHSH on density matrices ----------
def E_rho(rho: np.ndarray, a: float, b: float) -> float:
    MA = meas_op(a); MB = meas_op(b)
    O  = kron2(MA, MB)
    return float(np.trace(rho @ O).real)

def S_rho(rho: np.ndarray, a: float, ap: float, b: float, bp: float) -> float:
    return E_rho(rho,a,b) + E_rho(rho,a,bp) + E_rho(rho,ap,b) - E_rho(rho,ap,bp)

# ---------- Probability distributions & no-signalling check ----------
def pm_projectors(angle: float) -> tuple[np.ndarray,np.ndarray]:
    """
    Projectors for ± outcomes of M(angle).
    Using spectral decomposition: M = |+><+| - |-><-|
    """
    M = meas_op(angle)
    vals, vecs = LA.eigh(M)     # eigenvalues are +/-1
    # order them (+1 first)
    order = np.argsort(-vals.real)
    V = vecs[:, order]
    Pplus = V[:,[0]] @ V[:,[0]].conj().T
    Pminus= V[:,[1]] @ V[:,[1]].conj().T
    return Pplus, Pminus

def joint_probs(rho: np.ndarray, a: float, b: float) -> dict:
    """
    Return joint probabilities P(x,y) for x,y in {+1,-1} measuring M_A(a), M_B(b).
    """
    PpA, PmA = pm_projectors(a)
    PpB, PmB = pm_projectors(b)
    # Tensor projectors
    Ppp = kron2(PpA, PpB)
    Ppm = kron2(PpA, PmB)
    Pmp = kron2(PmA, PpB)
    Pmm = kron2(PmA, PmB)
    probs = {
        (+1,+1): float(np.trace(rho @ Ppp).real),
        (+1,-1): float(np.trace(rho @ Ppm).real),
        (-1,+1): float(np.trace(rho @ Pmp).real),
        (-1,-1): float(np.trace(rho @ Pmm).real),
    }
    return probs

def no_signalling_violation(rho: np.ndarray, a: float, a2: float, b: float, bp: float, tol: float=1e-8) -> dict:
    """
    Check P_A(+|a,b) == P_A(+|a,b') and P_B(+|a,b) == P_B(+|a',b) within tol.
    Returns the absolute diffs.
    """
    # A marginals
    P_ab  = joint_probs(rho,a,b)
    P_abp = joint_probs(rho,a,bp)
    PA_plus_b  = P_ab[(+1,+1)] + P_ab[(+1,-1)]
    PA_plus_bp = P_abp[(+1,+1)] + P_abp[(+1,-1)]
    # swap a→a2 to test B-marginal independence of a-setting
    P_a2b  = joint_probs(rho,a2,b)
    PB_plus_a  = P_ab[(+1,+1)] + P_ab[(-1,+1)]
    PB_plus_a2 = P_a2b[(+1,+1)] + P_a2b[(-1,+1)]
    return {
        "ΔA_marginal": abs(PA_plus_b - PA_plus_bp),
        "ΔB_marginal": abs(PB_plus_a - PB_plus_a2),
        "tol": tol,
    }

# ---------- Scans ----------
def scan_tau_heisenberg(J: float=1.0, taus=None, angles=None):
    """
    Prepare |01> and entangle with Heisenberg unitary U=exp(-iJτ(σ·σ)).
    Return S(τ) across taus.
    """
    if taus is None:
        taus = np.linspace(0, np.pi/2, 41)
    if angles is None:
        a, ap, b, bp = 0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi
    else:
        a, ap, b, bp = angles
    sx, sy, sz = pauli()
    H = kron2(sx,sx) + kron2(sy,sy) + kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0  # |01>
    Svals = []
    for tau in taus:
        U = expm(-1j * J * tau * H)
        psi = U @ psi0
        rho = ket_to_rho(psi)
        Svals.append(S_rho(rho,a,ap,b,bp))
    return np.array(taus), np.array(Svals)

def scan_noise_on_bell(p_list=None, angles=None):
    """
    Depolarize |Φ+> with strength p and compute S. Threshold should drop below 2 at ~p>~0.293.
    """
    if p_list is None:
        p_list = np.linspace(0, 0.5, 26)
    if angles is None:
        a, ap, b, bp = 0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi
    else:
        a, ap, b, bp = angles
    v = np.zeros(4, complex); v[0]=v[3]=1/np.sqrt(2)  # |Φ+>
    rho0 = ket_to_rho(v)
    out = []
    for p in p_list:
        rho = depolarize_rho(rho0, p)
        out.append(S_rho(rho,a,ap,b,bp))
    return np.array(p_list), np.array(out)

def correlation_tensor(rho: np.ndarray) -> np.ndarray:
    """Return 3x3 T with T_ij = Tr[rho (σ_i ⊗ σ_j)], i,j in {x,y,z}."""
    sx, sy, sz = pauli()
    sigs = [sx, sy, sz]
    T = np.zeros((3,3), float)
    for i,A in enumerate(sigs):
        for j,B in enumerate(sigs):
            T[i,j] = float(np.trace(rho @ kron2(A,B)).real)
    return T

def S_max_horodecki(rho: np.ndarray) -> float:
    """
    Horodecki formula: S_max = 2 * sqrt(m1 + m2),
    where m1>=m2 are the two largest eigenvalues of T^T T.
    """
    T = correlation_tensor(rho)
    M = T.T @ T
    evals = np.linalg.eigvalsh(M)
    m1, m2 = sorted(evals)[-2:]
    return float(2.0 * np.sqrt(m1 + m2))

def scan_tau_heisenberg_Smax(J: float=1.0, taus=None):
    """Like scan_tau_heisenberg, but returns S_max(τ)."""
    if taus is None:
        taus = np.linspace(0, np.pi/2, 41)
    sx, sy, sz = pauli()
    H = kron2(sx,sx) + kron2(sy,sy) + kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0  # |01>
    from scipy.linalg import expm
    Svals = []
    for tau in taus:
        U = expm(-1j * J * tau * H)
        psi = U @ psi0
        rho = ket_to_rho(psi)
        Svals.append(S_max_horodecki(rho))
    return np.array(taus), np.array(Svals)

def local_Z_phase_align(psi: np.ndarray) -> np.ndarray:
    """
    Apply local Z rotations to map c01 |01> + c10 |10> → (|01> + |10>)/√2 up to a global phase,
    without changing entanglement. Works when the state lives in span{|01>,|10>}.
    """
    psi = psi.copy()
    c01 = psi[1]; c10 = psi[2]
    if abs(c01) < 1e-12 or abs(c10) < 1e-12:
        return psi
    phi = np.angle(c10) - np.angle(c01)

    # Local Z on qubit B by -phi/2 and on qubit A by +phi/2 achieves relative phase cancellation.
    Z = np.array([[1,0],[0,-1]], complex)
    RzA = np.diag([np.exp(+1j*phi/2), np.exp(-1j*phi/2)])
    RzB = np.diag([np.exp(-1j*phi/2), np.exp(+1j*phi/2)])
    Uloc = np.kron(RzA, RzB)
    psi2 = Uloc @ psi
    # re-normalize just in case
    psi2 = psi2 / np.linalg.norm(psi2)
    return psi2


def correlation_tensor(rho: np.ndarray) -> np.ndarray:
    sx, sy, sz = pauli()
    sigs = [sx, sy, sz]
    T = np.zeros((3,3), float)
    for i,A in enumerate(sigs):
        for j,B in enumerate(sigs):
            T[i,j] = float(np.trace(rho @ kron2(A,B)).real)
    return T

def S_max_horodecki(rho: np.ndarray) -> float:
    T = correlation_tensor(rho)
    M = T.T @ T
    evals = np.linalg.eigvalsh(M)
    m1, m2 = sorted(evals)[-2:]
    return float(2.0 * np.sqrt(m1 + m2))

def frame_unitary_from_basis(u: np.ndarray) -> np.ndarray:
    """
    Build a 2x2 unitary W whose first column is u (|0_frame>) and second is an orthonormal u_perp.
    """
    u = u / (np.linalg.norm(u) + 1e-15)
    # orthonormal complement via J + Gram-Schmidt
    Jc = np.array([[0,-1],[1,0]], complex)
    w = (Jc @ u.reshape(2,1)).ravel()
    w = w - (np.vdot(u,w) * u)
    n = np.linalg.norm(w); w = w / (n + 1e-15)
    # phase-clean second column for determinantal unitary (optional)
    return np.column_stack([u, w])

def S_fixed_in_frames(psi: np.ndarray, a: float, ap: float, b: float, bp: float,
                      WA: np.ndarray, WB: np.ndarray) -> float:
    """
    Evaluate CHSH with fixed angles (a,a',b,b') but with measurement axes expressed
    in the local lattice frames via WA, WB (2x2 unitaries).
    """
    MA_a   = WA @ meas_op(a)  @ WA.conj().T
    MA_ap  = WA @ meas_op(ap) @ WA.conj().T
    MB_b   = WB @ meas_op(b)  @ WB.conj().T
    MB_bp  = WB @ meas_op(bp) @ WB.conj().T
    E_ab   = np.vdot(psi, kron2(MA_a,  MB_b ) @ psi).real
    E_abp  = np.vdot(psi, kron2(MA_a,  MB_bp) @ psi).real
    E_apb  = np.vdot(psi, kron2(MA_ap, MB_b ) @ psi).real
    E_apbp = np.vdot(psi, kron2(MA_ap, MB_bp) @ psi).real
    return float(E_ab + E_abp + E_apb - E_apbp)
