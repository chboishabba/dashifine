# ============================================
# runner_mixed_modulus.py
# ============================================
from __future__ import annotations
import argparse, numpy as np
import lattice_chsh as LCH
import chsh_extras  as CHEX
import chsh_harness as CHSH
from scipy.linalg import expm

def main():
    p = argparse.ArgumentParser(description="Mixed-modulus (mod 6 vs mod 9) lattice CHSH")
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--M_A", type=int, default=6, help="modulus for leg A")
    p.add_argument("--M_B", type=int, default=9, help="modulus for leg B")
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    p.add_argument("--tau", type=float, default=np.pi/8, help="entangler time")
    p.add_argument("--Jprep", type=float, default=1.0)
    p.add_argument("--angles", type=float, nargs=4, default=None, metavar=("a","ap","b","bp"),
                   help="optional fixed angles; if omitted we compute S_max")
    args = p.parse_args()

    wallA, wallB = args.NA//2, args.NB//2

    # Build two legs with different moduli
    HA, valsA, vecsA = LCH.build_single_leg_open_modulus(args.NA, args.t1, args.t2, wallA, args.M_A)
    HB, valsB, vecsB = LCH.build_single_leg_open_modulus(args.NB, args.t1, args.t2, wallB, args.M_B)

    # Extract wall-plane frames (2D) for reporting (optional)
    uA, _ = LCH.extract_wall_qubit_frame(vecsA, args.NA, wallA, which_block="A")
    uB, _ = LCH.extract_wall_qubit_frame(vecsB, args.NB, wallB, which_block="A")

    # Prepare a 2-qubit state with Heisenberg entangler (abstract qubit layer)
    # (You can also just use CHSH.bell_state_phi_plus() if you want the ideal Bell.)
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx) + CHEX.kron2(sy,sy) + CHEX.kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0  # |01>
    U = expm(-1j * args.Jprep * args.tau * H2)
    psi = U @ psi0
    rho = CHEX.ket_to_rho(psi)

    # Evaluate either S_max (basis independent) or S at fixed angles
    if args.angles is None:
        Smax = CHEX.S_max_horodecki(rho)
        print(f"S_max (Heisenberg τ={args.tau:.4f}) = {Smax:.6f}")
    else:
        a, ap, b, bp = args.angles
        S = CHEX.S_rho(rho, a, ap, b, bp)
        print(f"S (fixed angles) = {S:.6f}")

    print("Near-zero (first few) eigvals:")
    print(" A:", valsA[:6])
    print(" B:", valsB[:6])

if __name__ == "__main__":
    main()
