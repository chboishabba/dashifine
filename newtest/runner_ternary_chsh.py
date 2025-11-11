# runner_ternary_chsh.py
from __future__ import annotations
import numpy as np
from embed_chsh_ternary import bell_phi_plus, chsh_operators_qubit, expectation_in_embedded_CHSH, default_tsig_angles
from ternary_hilbert import embed_qubit_plane

def main():
    psi = bell_phi_plus()
    a,ap,b,bp = default_tsig_angles()
    S = chsh_operators_qubit(a,ap,b,bp)

    # Two different qubit planes inside the two qutrits (tweak angles if you like)
    WA = embed_qubit_plane(theta=0.0, phi=0.0)
    WB = embed_qubit_plane(theta=0.0, phi=0.0)

    Sval = expectation_in_embedded_CHSH(psi, S, WA, WB)
    print(f"Embedded CHSH <S> ≈ {Sval:.6f} (Tsirelson bound ~ 2.828427)")

if __name__=="__main__":
    main()
