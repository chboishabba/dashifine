# ============================================
# runner_cross_moduli_compare.py
# Compare M_A=6, M_B=9 vs M_A=9, M_B=6 with Δ heatmap
# Modes: (k, p_round) or (k, sigma)
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
    return float(np.mean(Svals))

def build_heatmap(NA, NB, M_A, M_B, t1, t2,
                  k_vals, y_vals, mode, seeds, psi, angles,
                  p_round_fixed, sigma_fixed):
    """
    Returns Z[j,i] = <S_fixed> at (y_vals[j], k_vals[i]) for chosen scan mode.
    """
    Z = np.zeros((len(y_vals), len(k_vals)))
    for j, y in enumerate(y_vals):
        for i, kq in enumerate(k_vals):
            if mode == "pround":
                p_round, sigma = float(y), float(sigma_fixed)
            else:  # mode == "sigma"
                p_round, sigma = float(p_round_fixed), float(y)
            Z[j, i] = compute_S_fixed_for_leg_pair(
                NA, NB, M_A, M_B, t1, t2,
                kq, p_round, sigma, seeds, psi, angles
            )
    return Z

def main():
    p = argparse.ArgumentParser(description="Cross-modulus asymmetry: compare (6,9) vs (9,6)")
    # Lattice & couplings
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)

    # Grids
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=36)
    p.add_argument("--kstep", type=int, default=1)

    p.add_argument("--mode", choices=["pround","sigma"], default="pround")
    p.add_argument("--p_round_min", type=float, default=0.0)
    p.add_argument("--p_round_max", type=float, default=1.0)
    p.add_argument("--p_round_pts", type=int, default=11)
    p.add_argument("--sigma_min", type=float, default=0.00)
    p.add_argument("--sigma_max", type=float, default=0.35)
    p.add_argument("--sigma_pts", type=int, default=12)

    # Fixed params for the alternate axis
    p.add_argument("--p_round_fixed", type=float, default=0.9)
    p.add_argument("--sigma_fixed", type=float, default=0.20)

    # State prep (Heisenberg entangler)
    p.add_argument("--tau", type=float, default=np.pi/8)
    p.add_argument("--Jprep", type=float, default=1.0)

    # Averaging
    p.add_argument("--seeds", type=int, default=16)

    # Output
    p.add_argument("--save", type=str, default=None)

    args = p.parse_args()

    # Prepare two-qubit state |ψ>
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx) + CHEX.kron2(sy,sy) + CHEX.kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0  # |01>
    U = expm(-1j * args.Jprep * args.tau * H2)
    psi = U @ psi0
    angles = (0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi)

    k_vals = np.arange(args.kmin, args.kmax+1, args.kstep)
    if args.mode == "pround":
        y_vals = np.linspace(args.p_round_min, args.p_round_max, args.p_round_pts)
        y_label = "p_round (quantization probability)"
    else:
        y_vals = np.linspace(args.sigma_min, args.sigma_max, args.sigma_pts)
        y_label = "σ (Gaussian phase stdev)"

    # Heatmaps for (6,9) and (9,6)
    print("Computing heatmap for M_A=6, M_B=9 ...")
    Z_69 = build_heatmap(args.NA, args.NB, 6, 9, args.t1, args.t2,
                         k_vals, y_vals, args.mode, args.seeds, psi, angles,
                         args.p_round_fixed, args.sigma_fixed)

    print("Computing heatmap for M_A=9, M_B=6 ...")
    Z_96 = build_heatmap(args.NA, args.NB, 9, 6, args.t1, args.t2,
                         k_vals, y_vals, args.mode, args.seeds, psi, angles,
                         args.p_round_fixed, args.sigma_fixed)

    Z_delta = Z_69 - Z_96

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axs[0].imshow(Z_69, aspect="auto", origin="lower",
                        extent=[k_vals[0], k_vals[-1], y_vals[0], y_vals[-1]])
    axs[0].set_title("⟨S_fixed⟩  (M_A=6, M_B=9)")
    axs[0].set_xlabel("Phase quantization k"); axs[0].set_ylabel(y_label)
    c0 = fig.colorbar(im0, ax=axs[0]); c0.set_label("mean over seeds")

    im1 = axs[1].imshow(Z_96, aspect="auto", origin="lower",
                        extent=[k_vals[0], k_vals[-1], y_vals[0], y_vals[-1]])
    axs[1].set_title("⟨S_fixed⟩  (M_A=9, M_B=6)")
    axs[1].set_xlabel("Phase quantization k"); axs[1].set_ylabel(y_label)
    c1 = fig.colorbar(im1, ax=axs[1]); c1.set_label("mean over seeds")

    # Symmetric color range for Δ
    vmax = np.max(np.abs(Z_delta))
    im2 = axs[2].imshow(Z_delta, aspect="auto", origin="lower",
                        extent=[k_vals[0], k_vals[-1], y_vals[0], y_vals[-1]],
                        vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axs[2].set_title("Δ = ⟨S⟩(6,9) − ⟨S⟩(9,6)")
    axs[2].set_xlabel("Phase quantization k"); axs[2].set_ylabel(y_label)
    c2 = fig.colorbar(im2, ax=axs[2]); c2.set_label("difference")

    fig.suptitle(f"Cross-modulus asymmetry  |  mode: {args.mode}", fontsize=12)
    if args.save:
        plt.savefig(args.save, dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    main()
