# ternary_hilbert.py
from __future__ import annotations
import numpy as np
import numpy.linalg as LA

# Root of unity for base-3 (qutrit)
def omega():
    return np.exp(2j*np.pi/3)

def basis_H3():
    """Return canonical qutrit basis |0>,|1>,|2> as columns (3x3 identity)."""
    return np.eye(3, dtype=complex)

def Z_qutrit():
    """Generalized Pauli Z: Z|k> = ω^k |k>."""
    ω = omega()
    return np.diag([1, ω, ω**2]).astype(complex)

def X_qutrit():
    """Generalized Pauli X: X|k> = |k+1 mod 3>."""
    X = np.zeros((3,3), complex)
    for k in range(3):
        X[(k+1)%3, k] = 1.0
    return X

def phase_form_matrix():
    """
    Optional 'phase pairing' Φ_ij = ω^{i*j}. This is NOT an inner product
    (it’s not positive-definite), but is useful as a mod-3 bilinear form
    for motif scoring/coarse-graining if desired.
    """
    ω = omega()
    Φ = np.zeros((3,3), complex)
    for i in range(3):
        for j in range(3):
            Φ[i,j] = ω**(i*j)
    return Φ

def embed_qubit_plane(theta: float=0.0, phi: float=0.0):
    """
    Return an isometry W: C^2 -> C^3 (3x2) selecting a qubit plane inside the qutrit.
    Default: span{|0>, |1>} with an SU(2) rotation on that plane.
    """
    # Start with |0>,|1> as columns in C^3
    W0 = np.array([[1,0],
                   [0,1],
                   [0,0]], dtype=complex)  # 3x2
    # Apply SU(2) on the columns (Bloch angles)
    # U2 = [cos θ/2, -e^{iφ} sin θ/2; e^{-iφ} sin θ/2, cos θ/2]
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    eip = np.exp(1j*phi)
    U2 = np.array([[c, -eip*s],
                   [np.conj(eip)*s, c]], dtype=complex)
    return W0 @ U2

def kron3(A, B, C):
    return np.kron(np.kron(A,B), C)

def modM_phase_rotation(M: int, r: int):
    """
    Return a 3x3 diagonal phase gate that rotates |k> by exp(2πi * r_k / M).
    Here we tie residues to basis labels k (simple prototype).
    """
    phases = [np.exp(2j*np.pi*((r*k) % M)/M) for k in range(3)]
    return np.diag(phases).astype(complex)
