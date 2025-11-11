# map_27_to_H3x3.py
from __future__ import annotations
import numpy as np
from motifs9 import idx_triplet_to_linear, coarse_grain_27_to9

def triplet_ket(i,j,k):
    """
    Return the 27-dim computational basis ket |i>⊗|j>⊗|k> as a length-27 vector.
    """
    v = np.zeros(27, complex)
    n = idx_triplet_to_linear(i,j,k)
    v[n] = 1.0
    return v

def mix_over_27(weights_27: np.ndarray):
    """
    Build a density vector over the 27 basis states (classical mixture).
    weights_27 must sum to 1; returns same vector.
    """
    w = np.asarray(weights_27, float).copy()
    s = w.sum()
    if s<=0: raise ValueError("weights must sum>0")
    w /= s
    return w

def coarse9_from_weights27(w27: np.ndarray):
    """Map a 27-probability vector to 9 motif probabilities."""
    return coarse_grain_27_to9(w27)
