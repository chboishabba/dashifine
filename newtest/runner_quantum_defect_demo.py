from __future__ import annotations
import numpy as np
from quantum_defect import rydberg_energy, alkali_level

def main():
    # Very rough, illustrative defects (not fitted): δ_s>δ_p>δ_d
    defects = {"s": 1.35, "p": 0.85, "d": 0.02}
    Zeff = 1.0  # effective charge for Rydberg electron

    print("n\tE_s\t\tE_p\t\tE_d   (Hartree)")
    for n in range(3, 9):
        Es = alkali_level(Zeff, n, defects["s"])
        Ep = alkali_level(Zeff, n, defects["p"])
        Ed = alkali_level(Zeff, n, defects["d"])
        print(f"{n}\t{Es: .6f}\t{Ep: .6f}\t{Ed: .6f}")

if __name__ == "__main__":
    main()
