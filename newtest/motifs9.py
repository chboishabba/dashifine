# motifs9.py
from __future__ import annotations
import numpy as np

def idx_triplet_to_linear(i,j,k):
    """Map (i,j,k) with i,j,k in {0,1,2} to 0..26."""
    return (i*3 + j)*3 + k

def linear_to_idx_triplet(n):
    i = n // 9
    j = (n % 9) // 3
    k = n % 3
    return i,j,k

def motifs9_bins():
    """
    Define 9 motif bins as (sum mod 3, number of zeros) pairs.
    This is a reasonable placeholder; adjust to your semantics M1..M9 later.
    Bins: (s, z) with s in {0,1,2}, z in {0,1,2}.
    """
    bins = []
    for s in [0,1,2]:
        for z in [0,1,2]:
            bins.append((s,z))
    return bins  # length 9

def triplet_to_motif(i,j,k):
    """
    Assign one of 9 motifs to a qutrit triplet index (i,j,k) ∈ {0,1,2}^3.
    Rule: motif = (sum mod 3, count of zeros) -> mapped to 0..8 in row-major.
    """
    s = (i + j + k) % 3
    z = (int(i==0) + int(j==0) + int(k==0))
    # row-major index: s∈{0,1,2}, z∈{0,1,2}
    return s*3 + z  # 0..8

def coarse_grain_27_to9(prob27: np.ndarray):
    """
    prob27: length-27 nonnegative array summing to 1 (occupancy over 27 backbone).
    Returns length-9 array summing to 1 (motif probabilities).
    """
    assert prob27.shape == (27,)
    out = np.zeros(9, float)
    for n in range(27):
        i,j,k = linear_to_idx_triplet(n)
        m = triplet_to_motif(i,j,k)
        out[m] += prob27[n]
    s = out.sum()
    if s > 0: out /= s
    return out

def projector_onto_motif(m: int):
    """
    Return a (27x27) diagonal projector selecting all triplets that fall into motif m.
    Useful if you lift operators from H3⊗3 and then coarse-grain expectations.
    """
    P = np.zeros((27,27), complex)
    for n in range(27):
        i,j,k = linear_to_idx_triplet(n)
        if triplet_to_motif(i,j,k) == m:
            P[n,n] = 1.0
    return P
