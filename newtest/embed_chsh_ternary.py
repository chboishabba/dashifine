# embed_chsh_ternary.py
from __future__ import annotations
import numpy as np
import numpy.linalg as LA
from ternary_hilbert import basis_H3, embed_qubit_plane, kron3

# Qubit Pauli matrices
SX = np.array([[0,1],[1,0]], complex)
SY = np.array([[0,-1j],[1j,0]], complex)
SZ = np.array([[1,0],[0,-1]], complex)

def qubit_rot(axis: str, angle: float):
    """Single-qubit rotation R_axis(angle) on a 2D qubit space."""
    if axis.lower()=='x':
        H = SX/2
    elif axis.lower()=='y':
        H = SY/2
    elif axis.lower()=='z':
        H = SZ/2
    else:
        raise ValueError("axis ∈ {x,y,z}")
    return LA.expm(-1j*angle*H)

def chsh_operators_qubit(a, ap, b, bp):
    """
    Return the standard CHSH operator on two qubits:
    A(a), A'(ap) on side A; B(b), B'(bp) on side B; each as ±1-valued observables.
    Use the Pauli frame in X-Z plane: A(θ)=cos θ Z + sin θ X with eigenvalues ±1.
    """
    def observable(theta):
        n = np.cos(theta)*SZ + np.sin(theta)*SX
        # Project onto ±1-valued observable by spectral sign
        w, V = LA.eigh(n)
        # numerical: clip near zeros
        sgn = np.sign(w.real)
        return V @ np.diag(sgn) @ V.conj().T

    A  = observable(a)
    Ap = observable(ap)
    B  = observable(b)
    Bp = observable(bp)
    # Build CHSH tensor operator: S = A⊗(B+B') + A'⊗(B-B')
    S = np.kron(A, (B + Bp)) + np.kron(Ap, (B - Bp))
    return S

def embed_two_qubits_in_two_qutrits(WA: np.ndarray, WB: np.ndarray):
    """
    WA, WB: isometries (3x2) selecting a qubit plane in each qutrit.
    Return Upull : C^4 -> C^9 isometry (product isometry) to lift 2-qubit ops/states into 2-qutrit space.
    """
    # Product isometry (3x2) ⊗ (3x2) = (9 x 4)
    return np.kron(WA, WB)

def expectation_in_embedded_CHSH(psi2: np.ndarray, S_qubit: np.ndarray, WA: np.ndarray, WB: np.ndarray):
    """
    psi2: 4-dim two-qubit statevector
    S_qubit: 4x4 CHSH operator in qubit space
    WA,WB: 3x2 isometries (qubit planes in qutrits)
    Returns: <S> evaluated after embedding into 3⊗3.
    """
    U = embed_two_qubits_in_two_qutrits(WA, WB)  # 9x4
    psi9 = U @ psi2                    # two-qutrit state in the 9-dim product
    S9   = U @ S_qubit @ U.conj().T    # pull S up
    num  = psi9.conj().T @ (S9 @ psi9)
    return float(np.real(num))

def default_tsig_angles():
    """Tsirelson-optimal angles (in the qubit model) for |Φ+>."""
    a   = 0.0
    ap  = 0.5*np.pi
    b   = 0.25*np.pi
    bp  = -0.25*np.pi
    return a,ap,b,bp

def bell_phi_plus():
    """Two-qubit |Φ+> state."""
    psi = np.zeros(4, complex)
    psi[0] = 1/np.sqrt(2)
    psi[3] = 1/np.sqrt(2)
    return psi
