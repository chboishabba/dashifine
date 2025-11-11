from __future__ import annotations
import numpy as np
from hydrogenic_numerov import numerov_bound_states

def main():
    # Z sweep; s and p channels
    for Z in [1, 2, 5, 10]:
        for l in [0, 1]:
            E, _, _ = numerov_bound_states(Z=Z, l=l, n_states=4)
            print(f"Z={Z:2d} l={l}  E ≈ {np.array2string(E, precision=6)}")
    print("Analytic H (Z=1, l=0): n=1,2,3 → [-0.5, -0.125, -0.055556] Hartree")

if __name__ == "__main__":
    main()
