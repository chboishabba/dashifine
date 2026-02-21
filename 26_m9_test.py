import numpy as np
from itertools import product

# ----------------------------
# Trit encoding
# ----------------------------

PLUS  = 1
ZERO  = 0
MINUS = -1

TRITS = [PLUS, ZERO, MINUS]

# 3 lenses × 3 time columns
# index layout: (lens, time)
# lens: 0=Self, 1=Norm, 2=Mirror
# time: 0=past,1=now,2=future

# ----------------------------
# Distance (Hamming)
# ----------------------------

def hamming(S, T):
    return np.sum(S != T)

# ----------------------------
# Motif classifier Π
# (minimal demonstration rules)
# ----------------------------

def motif(S):
    b0 = S[:,1]  # present column

    # M1 Robust-Allow
    if np.all(b0 == PLUS):
        return 1

    # M9 All-red spine
    if np.all(b0 == MINUS):
        return 9

    # M3 Role-Gated
    if S[0,1]==PLUS and S[2,1]==PLUS and S[1,1]==MINUS:
        return 3

    # fallback
    return 5  # undecidable buffer

# ----------------------------
# Canonical representatives
# ----------------------------

def canonical(m):
    S = np.zeros((3,3),dtype=int)

    if m == 1:
        S[:,1] = PLUS
    elif m == 9:
        S[:,1] = MINUS
    elif m == 3:
        S[0,1] = PLUS
        S[2,1] = PLUS
        S[1,1] = MINUS
    else:
        S[:,1] = ZERO

    return S

# ----------------------------
# Projection K
# ----------------------------

def K(S):
    m = motif(S)
    return canonical(m)

# ----------------------------
# Energy
# ----------------------------

def energy(S):
    return hamming(S, canonical(motif(S)))

# ----------------------------
# Contraction check
# ----------------------------

def test_contraction():
    states = []
    for values in product(TRITS, repeat=9):
        S = np.array(values).reshape(3,3)
        states.append(S)

    for S in states[:200]:
        for T in states[:200]:
            d0 = hamming(S,T)
            d1 = hamming(K(S), K(T))
            if d1 > d0:
                return False
    return True

# ----------------------------
# Strict decrease test
# ----------------------------

def test_strict_decrease():
    for values in product(TRITS, repeat=9):
        S = np.array(values).reshape(3,3)
        if not np.array_equal(K(S), S):
            if energy(K(S)) >= energy(S):
                return False
    return True

if __name__ == "__main__":
    print("Contractive:", test_contraction())
    print("Strict energy decrease:", test_strict_decrease())
