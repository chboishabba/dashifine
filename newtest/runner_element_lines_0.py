from __future__ import annotations
import numpy as np
from hydrogenic_numerov import numerov_bound_states

# Hartree to eV and to wavelength (nm)
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_HZ = 6.579683920711e15  # 1 Ha / h
C_NM_S = 2.99792458e17

def transitions_from_levels(E):
    """
    Given array of energies (Hartree), return all downward transitions (Ei->Ej, i<j),
    with ΔE (eV), freq (Hz), wavelength (nm).
    """
    lines = []
    for i in range(len(E)):
        for j in range(i+1, len(E)):
            dE = E[j] - E[i]  # note: bound levels E<0, so j>i is less bound (closer to 0)
            if dE > 0:
                eV = dE * HARTREE_TO_EV
                Hz = dE * HARTREE_TO_HZ
                nm = C_NM_S / Hz
                lines.append((i, j, float(dE), float(eV), float(Hz), float(nm)))
    return sorted(lines, key=lambda t: t[3], reverse=True)

def main():
    Z = 1
    levels_per_l = {}
    for l in [0, 1, 2]:
        E, _, _ = numerov_bound_states(Z=Z, l=l, n_states=5)
        levels_per_l[l] = E
        print(f"Z={Z} l={l} levels (Ha): {np.array2string(E, precision=6)}")

    # collect a small combined list (mix s/p/d) and form transitions
    all_levels = np.sort(np.concatenate(list(levels_per_l.values())))
    lines = transitions_from_levels(all_levels[:8])  # first 8 levels only for brevity

    # save CSV
    import csv
    with open("lines_Z1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i","j","dE_Ha","dE_eV","freq_Hz","lambda_nm"])
        w.writerows(lines)
    print("Wrote lines_Z1.csv with", len(lines), "transitions")

if __name__ == "__main__":
    main()
