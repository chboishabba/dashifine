# ============================================
# runner_overlay_decoherence.py
# Overlay: quantized rounding (mod-grain) vs Gaussian jitter vs depolarizing
# ============================================
from __future__ import annotations
import argparse, numpy as np
import lattice_chsh as LCH
import chsh_extras  as CHEX
from scipy.linalg import expm
import matplotlib.pyplot as plt

def scan_channel(
    NA, NB, M_A, M_B, t1, t2, k_values, seeds,
    make_leg_A, make_leg_B,
    psi, angles
):
    """Generic scanner that returns k_list, mean_list, std_list for S_fixed."""
    a, ap, b, bp = angles
    k_list, mean_list, std_list = [], [], []
    wallA, wallB = NA//2, NB//2

    for kval in k_values:
        Svals=[]
        for s in range(seeds):
            rng = np.random.default_rng(s)
            HA, valsA, vecsA = make_leg_A(NA, t1, t2, wallA, M_A, kval, rng)
            HB, valsB, vecsB = make_leg_B(NB, t1, t2, wallB, M_B, kval, rng)

            uA,_ = LCH.extract_wall_qubit_frame(vecsA, NA, wallA, which_block="A")
            uB,_ = LCH.extract_wall_qubit_frame(vecsB, NB, wallB, which_block="A")
            WA, WB = CHEX.frame_unitary_from_basis(uA), CHEX.frame_unitary_from_basis(uB)

            Svals.append(CHEX.S_fixed_in_frames(psi, a, ap, b, bp, WA, WB))
        Svals = np.array(Svals)
        k_list.append(kval)
        mean_list.append(Svals.mean())
        std_list.append(Svals.std())
    return np.array(k_list), np.array(mean_list), np.array(std_list)

def main():
    p = argparse.ArgumentParser(description="Overlay modular quantization vs Gaussian vs depolarizing")
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--M_A", type=int, default=6)
    p.add_argument("--M_B", type=int, default=9)
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=36)
    p.add_argument("--kstep", type=int, default=1)
    p.add_argument("--seeds", type=int, default=32)

    # Quantized rounding controls
    p.add_argument("--p_round", type=float, default=0.9, help="probability to round a given edge (quantized)")

    # Gaussian jitter controls (phase sigma in radians)
    p.add_argument("--sigma", type=float, default=0.20, help="Gaussian phase stdev for continuous jitter")

    # State prep (Heisenberg entangler)
    p.add_argument("--tau", type=float, default=np.pi/8)
    p.add_argument("--Jprep", type=float, default=1.0)

    # Depolarizing reference
    p.add_argument("--dep_p", type=float, default=0.25)

    # Output
    p.add_argument("--save", type=str, default=None, help="optional path to save figure (e.g., overlay.png)")

    args = p.parse_args()

    # Prepare a fixed two-qubit state ψ
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx) + CHEX.kron2(sy,sy) + CHEX.kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0  # |01>
    U = expm(-1j * args.Jprep * args.tau * H2)
    psi = U @ psi0

    # Fixed Tsirelson angles (expressed in local frames by S_fixed_in_frames)
    angles = (0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi)

    # k grid
    k_values = np.arange(args.kmin, args.kmax + 1, args.kstep)

    # --- Channel 1: quantized rounding (mod-grain)
    def makeA_quant(N,t1,t2,wall,M,k,rng):
        return LCH.build_single_leg_open_modulus_quantized_random(
            N, t1, t2, wall, M, k_quant=k, p_round=args.p_round, rng=rng
        )
    def makeB_quant(N,t1,t2,wall,M,k,rng):
        return LCH.build_single_leg_open_modulus_quantized_random(
            N, t1, t2, wall, M, k_quant=k, p_round=args.p_round, rng=rng
        )
    k_q, m_q, s_q = scan_channel(
        args.NA, args.NB, args.M_A, args.M_B, args.t1, args.t2,
        k_values, args.seeds, makeA_quant, makeB_quant, psi, angles
    )

    # --- Channel 2: Gaussian phase jitter (continuous)
    def makeA_gauss(N,t1,t2,wall,M,k,rng):
        # We reuse 'k' as an index just to align grids; it doesn’t affect gaussian directly.
        return LCH.build_single_leg_open_modulus_gaussian(
            N, t1, t2, wall, M, sigma=args.sigma, rng=rng
        )
    def makeB_gauss(N,t1,t2,wall,M,k,rng):
        return LCH.build_single_leg_open_modulus_gaussian(
            N, t1, t2, wall, M, sigma=args.sigma, rng=rng
        )
    k_g, m_g, s_g = scan_channel(
        args.NA, args.NB, args.M_A, args.M_B, args.t1, args.t2,
        k_values, args.seeds, makeA_gauss, makeB_gauss, psi, angles
    )

    # --- Depolarizing (flat reference)
    rho0 = CHEX.ket_to_rho(psi)
    S_dep = CHEX.S_rho(CHEX.depolarize_rho(rho0, args.dep_p), *angles)

    # --- Plot
    plt.figure(figsize=(9,5))
    # quantized
    plt.errorbar(k_q, m_q, yerr=s_q, marker='o', capsize=3, label=f'Quantized rounding (p_round={args.p_round})')
    # gaussian
    plt.errorbar(k_g, m_g, yerr=s_g, marker='s', capsize=3, label=f'Gaussian jitter (σ={args.sigma})')
    # baselines
    plt.axhline(2.0,    ls='--', label='Classical bound')
    plt.axhline(2.8284, ls=':',  label='Tsirelson')
    plt.axhline(S_dep,  ls='-.', label=f'Depolarizing (p={args.dep_p:.2f})')

    plt.xlabel("Phase quantization k")
    plt.ylabel("<S_fixed> ± σ")
    plt.title("Modular quantization vs Gaussian jitter vs depolarizing")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    main()
