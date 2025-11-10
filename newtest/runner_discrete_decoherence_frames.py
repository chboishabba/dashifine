# ============================================
# runner_discrete_decoherence_frames.py
# Tie CHSH angles to lattice wall-mode frames & scan phase quantization k
# ============================================
from __future__ import annotations
import argparse, numpy as np
import lattice_chsh as LCH
import chsh_extras  as CHEX
from scipy.linalg import expm

def main():
    p = argparse.ArgumentParser(description="Discrete decoherence with lattice-constrained CHSH axes")
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--M_A", type=int, default=6,  help="modulus for leg A")
    p.add_argument("--M_B", type=int, default=6,  help="modulus for leg B")
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=24)
    p.add_argument("--kstep", type=int, default=3)
    p.add_argument("--tau", type=float, default=np.pi/8)
    p.add_argument("--Jprep", type=float, default=1.0)
    # fixed Tsirelson angles (can change to taste)
    p.add_argument("--angles", type=float, nargs=4, default=[0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi],
                   metavar=("a","ap","b","bp"))
    args = p.parse_args()

    wallA, wallB = args.NA//2, args.NB//2

    # Prepare a fixed two-qubit state via Heisenberg entangler (you can swap to ideal Bell)
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx) + CHEX.kron2(sy,sy) + CHEX.kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0  # |01>
    U = expm(-1j * args.Jprep * args.tau * H2)
    psi = U @ psi0

    a, ap, b, bp = args.angles
    print("k\tS_fixed\t||uA|| ||uB||\t(first eigs)")

    for k in range(args.kmin, args.kmax+1, args.kstep):
        # Build legs with modulus AND quantized inter-edge phases (2π/k)
        HA, valsA, vecsA = LCH.build_single_leg_open_modulus_quantized(args.NA, args.t1, args.t2, wallA, args.M_A, k)
        HB, valsB, vecsB = LCH.build_single_leg_open_modulus_quantized(args.NB, args.t1, args.t2, wallB, args.M_B, k)

        # Extract local wall-mode 2D directions (A/B), build frame unitaries
        uA, _ = LCH.extract_wall_qubit_frame(vecsA, args.NA, wallA, which_block="A")
        uB, _ = LCH.extract_wall_qubit_frame(vecsB, args.NB, wallB, which_block="A")
        WA = CHEX.frame_unitary_from_basis(uA)
        WB = CHEX.frame_unitary_from_basis(uB)

        # CHSH with FIXED angles in those frames
        S_fixed = CHEX.S_fixed_in_frames(psi, a, ap, b, bp, WA, WB)

        print(f"{k}\t{S_fixed:.6f}\t{np.linalg.norm(uA):.3f} {np.linalg.norm(uB):.3f}\t{valsA[:2]} {valsB[:2]}")

if __name__ == "__main__":
    main()
