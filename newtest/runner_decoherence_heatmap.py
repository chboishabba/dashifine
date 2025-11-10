# ============================================
# runner_decoherence_heatmap.py
# Heatmaps: <S_fixed> vs (k, sigma)  OR  vs (k, p_round)
# ============================================
from __future__ import annotations
import argparse, numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

import lattice_chsh as LCH
import chsh_extras  as CHEX

def compute_S_fixed_for_leg_pair(NA, NB, M_A, M_B, t1, t2,
                                 k_quant, p_round, sigma, seeds,
                                 psi, angles):
    """Average S_fixed over RNG seeds for one (k, p_round, sigma) tuple."""
    wallA, wallB = NA//2, NB//2
    a, ap, b, bp = angles
    Svals = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        HA, valsA, vecsA = LCH.build_single_leg_open_modulus_quantized_then_gaussian(
            NA, t1, t2, wallA, M_A, k_quant, p_round, sigma, rng=rng)
        HB, valsB, vecsB = LCH.build_single_leg_open_modulus_quantized_then_gaussian(
            NB, t1, t2, wallB, M_B, k_quant, p_round, sigma, rng=rng)
        uA,_ = LCH.extract_wall_qubit_frame(vecsA, NA, wallA, which_block="A")
        uB,_ = LCH.extract_wall_qubit_frame(vecsB, NB, wallB, which_block="A")
        WA, WB = CHEX.frame_unitary_from_basis(uA), CHEX.frame_unitary_from_basis(uB)
        Svals.append(CHEX.S_fixed_in_frames(psi, a, ap, b, bp, WA, WB))
    return float(np.mean(Svals)), float(np.std(Svals))

def main():
    p = argparse.ArgumentParser(description="Heatmaps of <S_fixed> vs (k, sigma) or vs (k, p_round)")
    # Lattice & moduli
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--M_A", type=int, default=6)
    p.add_argument("--M_B", type=int, default=9)
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    # Grids
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=36)
    p.add_argument("--kstep", type=int, default=1)
    p.add_argument("--sigma_min", type=float, default=0.00)
    p.add_argument("--sigma_max", type=float, default=0.35)
    p.add_argument("--sigma_pts", type=int, default=12)
    p.add_argument("--p_round_min", type=float, default=0.0)
    p.add_argument("--p_round_max", type=float, default=1.0)
    p.add_argument("--p_round_pts", type=int, default=11)
    # Seeds, state prep, mode
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--tau", type=float, default=np.pi/8)
    p.add_argument("--Jprep", type=float, default=1.0)
    p.add_argument("--mode", choices=["sigma","pround"], default="sigma",
                   help="sigma: heatmap vs (k, sigma) at fixed p_round; pround: vs (k, p_round) at fixed sigma")
    p.add_argument("--p_round_fixed", type=float, default=0.9)
    p.add_argument("--sigma_fixed", type=float, default=0.20)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    # Prepare two-qubit state |ψ> via Heisenberg entangler
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx) + CHEX.kron2(sy,sy) + CHEX.kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0   # |01>
    U = expm(-1j * args.Jprep * args.tau * H2)
    psi = U @ psi0
    angles = (0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi)

    # Axes
    k_vals = np.arange(args.kmin, args.kmax+1, args.kstep)

    if args.mode == "sigma":
        y_vals = np.linspace(args.sigma_min, args.sigma_max, args.sigma_pts)
        label_y = "σ (Gaussian phase stdev)"
        p_round = args.p_round_fixed
        Z = np.zeros((len(y_vals), len(k_vals)))
        for j, sigma in enumerate(y_vals):
            for i, kq in enumerate(k_vals):
                meanS, _ = compute_S_fixed_for_leg_pair(
                    args.NA, args.NB, args.M_A, args.M_B, args.t1, args.t2,
                    kq, p_round, sigma, args.seeds, psi, angles
                )
                Z[j, i] = meanS

    else:  # mode == "pround"
        y_vals = np.linspace(args.p_round_min, args.p_round_max, args.p_round_pts)
        label_y = "p_round (quantization probability)"
        sigma = args.sigma_fixed
        Z = np.zeros((len(y_vals), len(k_vals)))
        for j, p_round in enumerate(y_vals):
            for i, kq in enumerate(k_vals):
                meanS, _ = compute_S_fixed_for_leg_pair(
                    args.NA, args.NB, args.M_A, args.M_B, args.t1, args.t2,
                    kq, p_round, sigma, args.seeds, psi, angles
                )
                Z[j, i] = meanS

    # Plot
    plt.figure(figsize=(10, 5))
    im = plt.imshow(Z, aspect="auto", origin="lower",
                    extent=[k_vals[0], k_vals[-1], y_vals[0], y_vals[-1]])
    cbar = plt.colorbar(im)
    cbar.set_label("<S_fixed> (mean over seeds)")
    plt.xlabel("Phase quantization k")
    plt.ylabel(label_y)
    plt.title(f"Heatmap of <S_fixed> vs (k, {label_y.split()[0]})   M_A={args.M_A}, M_B={args.M_B}")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    main()
