# ============================================
# runner_composite_moduli.py
# Composite moduli (e.g., Ms_A=[3,5], Ms_B=[9]) vs swap, with Δ map
# Modes: (k, p_round) or (k, sigma)
# ============================================
from __future__ import annotations
import argparse, numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

import lattice_chsh as LCH
import chsh_extras  as CHEX

def compute_S_fixed_pair(
    NA, NB, Ms_A, Ms_B, t1, t2,
    k_quant, p_round, sigma, seeds,
    psi, angles, *,
    mode: str = "lcm", weights_A=None, weights_B=None
):
    wallA, wallB = NA//2, NB//2
    a, ap, b, bp = angles
    Svals = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        HA, vA, eA = LCH.build_single_leg_open_composite_quantized_then_gaussian(
            NA, t1, t2, wallA, Ms_A, k_quant, p_round, sigma,
            mode=mode, weights=weights_A, rng=rng
        )
        HB, vB, eB = LCH.build_single_leg_open_composite_quantized_then_gaussian(
            NB, t1, t2, wallB, Ms_B, k_quant, p_round, sigma,
            mode=mode, weights=weights_B, rng=rng
        )
        uA,_ = LCH.extract_wall_qubit_frame(eA, NA, wallA, which_block="A")
        uB,_ = LCH.extract_wall_qubit_frame(eB, NB, wallB, which_block="A")
        WA, WB = CHEX.frame_unitary_from_basis(uA), CHEX.frame_unitary_from_basis(uB)
        Svals.append(CHEX.S_fixed_in_frames(psi, a, ap, b, bp, WA, WB))
    return float(np.mean(Svals))

def build_heatmap(
    NA, NB, Ms_A, Ms_B, t1, t2, k_vals, y_vals, scan_mode,
    seeds, psi, angles, *,
    p_round_fixed, sigma_fixed, comp_mode="lcm", weights_A=None, weights_B=None
):
    Z = np.zeros((len(y_vals), len(k_vals)))
    for j, y in enumerate(y_vals):
        for i, kq in enumerate(k_vals):
            if scan_mode == "pround":
                p_round, sigma = float(y), float(sigma_fixed)
            else:
                p_round, sigma = float(p_round_fixed), float(y)
            Z[j, i] = compute_S_fixed_pair(
                NA, NB, Ms_A, Ms_B, t1, t2, kq, p_round, sigma, seeds, psi, angles,
                mode=comp_mode, weights_A=weights_A, weights_B=weights_B
            )
    return Z

def main():
    p = argparse.ArgumentParser(description="Composite moduli compare (Ms_A vs Ms_B) and swap")
    # Lattice & couplings
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)

    # Composite moduli (comma-separated lists)
    p.add_argument("--Ms_A", type=str, default="3,5")
    p.add_argument("--Ms_B", type=str, default="9")
    p.add_argument("--comp_mode", choices=["lcm","sum"], default="lcm",
                   help="How to combine Ms into phases")

    # Grids
    p.add_argument("--mode", choices=["pround","sigma"], default="pround")
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=45)   # extend to see LCM(15,9)=45
    p.add_argument("--kstep", type=int, default=1)
    p.add_argument("--p_round_min", type=float, default=0.0)
    p.add_argument("--p_round_max", type=float, default=1.0)
    p.add_argument("--p_round_pts", type=int, default=11)
    p.add_argument("--sigma_min", type=float, default=0.00)
    p.add_argument("--sigma_max", type=float, default=0.35)
    p.add_argument("--sigma_pts", type=int, default=12)

    # Fixed params
    p.add_argument("--p_round_fixed", type=float, default=0.9)
    p.add_argument("--sigma_fixed", type=float, default=0.20)

    # State prep
    p.add_argument("--tau", type=float, default=np.pi/8)
    p.add_argument("--Jprep", type=float, default=1.0)

    # Averaging
    p.add_argument("--seeds", type=int, default=16)

    # Output
    p.add_argument("--save", type=str, default=None)

    args = p.parse_args()

    Ms_A = [int(x) for x in args.Ms_A.split(",") if x.strip()]
    Ms_B = [int(x) for x in args.Ms_B.split(",") if x.strip()]

    # Prepare |ψ>
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx) + CHEX.kron2(sy,sy) + CHEX.kron2(sz,sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0
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

    print(f"Computing Ms_A={Ms_A}, Ms_B={Ms_B} ...")
    Z_AB = build_heatmap(
        args.NA, args.NB, Ms_A, Ms_B, args.t1, args.t2, k_vals, y_vals, args.mode,
        args.seeds, psi, angles,
        p_round_fixed=args.p_round_fixed, sigma_fixed=args.sigma_fixed,
        comp_mode=args.comp_mode
    )
    print(f"Computing Ms_A={Ms_B}, Ms_B={Ms_A} (swap) ...")
    Z_BA = build_heatmap(
        args.NA, args.NB, Ms_B, Ms_A, args.t1, args.t2, k_vals, y_vals, args.mode,
        args.seeds, psi, angles,
        p_round_fixed=args.p_round_fixed, sigma_fixed=args.sigma_fixed,
        comp_mode=args.comp_mode
    )
    Z_delta = Z_AB - Z_BA

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im0 = axs[0].imshow(Z_AB, aspect="auto", origin="lower",
                        extent=[k_vals[0], k_vals[-1], y_vals[0], y_vals[-1]])
    axs[0].set_title(f"⟨S_fixed⟩  Ms_A={Ms_A}  Ms_B={Ms_B}")
    axs[0].set_xlabel("Phase quantization k"); axs[0].set_ylabel(y_label)
    c0 = fig.colorbar(im0, ax=axs[0]); c0.set_label("mean over seeds")

    im1 = axs[1].imshow(Z_BA, aspect="auto", origin="lower",
                        extent=[k_vals[0], k_vals[-1], y_vals[0], y_vals[-1]])
    axs[1].set_title(f"⟨S_fixed⟩  Ms_A={Ms_B}  Ms_B={Ms_A}")
    axs[1].set_xlabel("Phase quantization k"); axs[1].set_ylabel(y_label)
    c1 = fig.colorbar(im1, ax=axs[1]); c1.set_label("mean over seeds")

    vmax = np.max(np.abs(Z_delta))
    im2 = axs[2].imshow(Z_delta, aspect="auto", origin="lower",
                        extent=[k_vals[0], k_vals[-1], y_vals[0], y_vals[-1]],
                        vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axs[2].set_title("Δ = ⟨S⟩(Ms_A,Ms_B) − ⟨S⟩(swap)")
    axs[2].set_xlabel("Phase quantization k"); axs[2].set_ylabel(y_label)
    c2 = fig.colorbar(im2, ax=axs[2]); c2.set_label("difference")

    fig.suptitle(f"Composite moduli ({args.comp_mode}) | mode: {args.mode}", fontsize=12)
    if args.save:
        plt.savefig(args.save, dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    main()
