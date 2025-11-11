# runner_element_lines.py
from __future__ import annotations
import numpy as np, csv
from hydrogenic_numerov import numerov_bound_states

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_HZ = 6.579683920711e15
C_NM_S = 2.99792458e17

def pair_lines(E_from, E_to, lfrom, lto):
    # all downward transitions from each level in lfrom to each level in lto
    lines = []
    for i, Ei in enumerate(E_from):
        for j, Ej in enumerate(E_to):
            dE = Ej - Ei  # Ej closer to 0 (less negative) => positive
            if dE > 0:
                eV = dE * HARTREE_TO_EV
                Hz = dE * HARTREE_TO_HZ
                nm = C_NM_S / Hz
                lines.append((lfrom, i, lto, j, float(dE), float(eV), float(Hz), float(nm)))
    return lines

def main():
    Z = 1
    levels = {}
    for l in [0,1,2]:
        E, _, _ = numerov_bound_states(Z=Z, l=l, n_states=6)
        levels[l] = np.sort(E)
        print(f"Z={Z} l={l} levels (Ha): {np.array2string(levels[l], precision=6)}")

    # Dipole selection: Δl = ±1 only
    lines = []
    lines += pair_lines(levels[0], levels[1], 0, 1)  # s -> p
    lines += pair_lines(levels[1], levels[0], 1, 0)  # p -> s
    lines += pair_lines(levels[1], levels[2], 1, 2)  # p -> d
    lines += pair_lines(levels[2], levels[1], 2, 1)  # d -> p

    # sort by energy (eV), high to low
    lines.sort(key=lambda t: t[5], reverse=True)

    with open("lines_Z1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["l_from","i","l_to","j","dE_Ha","dE_eV","freq_Hz","lambda_nm"])
        w.writerows(lines)
    print("Wrote lines_Z1.csv with", len(lines), "dipole-allowed transitions.")
    print("Top 8 lines (eV, nm):")
    for row in lines[:8]:
        print(f"ΔE={row[5]:7.3f} eV  λ={row[7]:9.2f} nm   (l{row[0]}[{row[1]}]→l{row[2]}[{row[3]}])")

if __name__ == "__main__":
    main()
