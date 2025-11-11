# hydrogenic.py
from __future__ import annotations
import numpy as np
import numpy.linalg as LA

def radial_grid(nr: int=800, rmax: float=200.0):
    r = np.linspace(1e-6, rmax, nr)  # avoid r=0 to prevent blowup
    dr = r[1]-r[0]
    return r, dr

def kinetic_fd(nr: int, dr: float, mass: float=1.0):
    """
    - (1/2m) d^2/dr^2 in atomic units (ħ=1).
    Simple Dirichlet-like second-difference on [0,rmax].
    """
    main = np.full(nr, -2.0)
    off  = np.ones(nr-1)
    D2   = (np.diag(main) + np.diag(off,1) + np.diag(off,-1)) / (dr*dr)
    T    = -(1.0/(2.0*mass)) * D2
    return T

def coulomb_potential(r: np.ndarray, Z: float):
    """V(r) = -Z/r (atomic units)."""
    return -Z / r

def centrifugal_l(l: int, r: np.ndarray, mass: float=1.0):
    """l(l+1)/(2 m r^2) term in atomic units."""
    return 0.5 * l*(l+1) / (mass * r**2)

def hydrogenic_spectrum(Z: float=1.0, l: int=0, nr: int=800, rmax: float=200.0, mass: float=1.0, n_eigs: int=6):
    """
    Return (energies, eigenvectors, r) for a simple 1e radial problem (fixed l).
    Coarse toy for Z-splitting experiments; not a full atomic solver.
    """
    r, dr = radial_grid(nr, rmax)
    T = kinetic_fd(nr, dr, mass)
    V = np.diag(coulomb_potential(r, Z) + centrifugal_l(l, r, mass))
    H = T + V
    vals, vecs = LA.eigh(H)
    # Sort lowest energies (bound states are negative)
    order = np.argsort(vals)
    vals  = vals[order][:n_eigs]
    vecs  = vecs[:,order][:,:n_eigs]
    # Normalize radial wavefunctions (∫|u|^2 dr = 1)
    for k in range(n_eigs):
        norm = np.sqrt(np.trapz(np.abs(vecs[:,k])**2, r))
        if norm>0: vecs[:,k] /= norm
    return vals, vecs, r
