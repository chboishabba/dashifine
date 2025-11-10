# ============================================
# runner_discrete_decoherence.py
# ============================================
from __future__ import annotations
import argparse, numpy as np
import lattice_chsh as LCH
import chsh_extras  as CHEX
from scipy.linalg import expm

def main():
    p = argparse.ArgumentParser(description="Discrete decoherence: phase quantization scan")
    p.add_argument("--N", type=int, default=41)
    p.add_argument("--M", type=int, default=6, help="modulus for lattice phases")
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    p.add_argument("--kmin", type=int, default=4)
    p.add_argument("--kmax", type=int, default=24)
    p.add_argument("--kstep", type=int, default=2)
    p.add_argument("--tau", type=float, default=np.pi/8)
    p.add_argument("--Jprep", type=float, default=1.0)
    args = p.parse_args()

    wall = args.N // 2

    # Prepare same 2-qubit Heisenberg state for all scans (so changes come from lattice/measurement frames)
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx) + CHEX.kron2(sy,sy) + CHEX.kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0
    U = expm(-1j * args.Jprep * args.tau * H2)
    psi = U @ psi0
    rho = CHEX.ket_to_rho(psi)

    print("k\tS_max\t(first 4 eigs)")
    for k in range(args.kmin, args.kmax+1, args.kstep):
        # Build a single leg with quantized phases (you can also do two legs and compare)
        H, vals, vecs = LCH.build_single_leg_open_modulus_quantized(args.N, args.t1, args.t2, wall, args.M, k)
        # We’re not using the frames to rotate axes here; we’re probing whether the substrate quantization
        # affects S_max if you later tie frames to lattice (extend as needed).
        Smax = CHEX.S_max_horodecki(rho)
        print(f"{k}\t{Smax:.6f}\t{vals[:4]}")

if __name__ == "__main__":
    main()
