# ============================================
# runner_phase_locked_entanglers.py
# Phase-locked entanglers: scan τ and/or (k, τ)
# Works with composite-moduli legs (e.g., Ms_A=[3,5], Ms_B=[9])
# ============================================
from __future__ import annotations
import argparse, numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

import lattice_chsh as LCH
import chsh_extras  as CHEX

# ----------------------------
# Core CHSH scan helpers
# ----------------------------

def prep_bell_like_state(tau: float, Jprep: float = 1.0) -> np.ndarray:
    """Prepare |ψ> = exp(-i Jprep * tau * (σx⊗σx + σy⊗σy + σz⊗σz)) |01>."""
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx, sx) + CHEX.kron2(sy, sy) + CHEX.kron2(sz, sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0
    U = expm(-1j * Jprep * tau * H2)
    return U @ psi0

def S_fixed_for_legs(
    NA: int, NB: int,
    Ms_A: list[int], Ms_B: list[int],
    t1: float, t2: float,
    k_quant: int, p_round: float, sigma: float,
    seeds: int,
    psi: np.ndarray,
    angles: tuple[float, float, float, float],
    *,
    comp_mode: str = "lcm",
    weights_A = None, weights_B = None,
) -> float:
    """Average S_fixed across RNG seeds for given legs and |ψ>."""
    wallA, wallB = NA // 2, NB // 2
    a, ap, b, bp = angles
    Svals = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        HA, vA, eA = LCH.build_single_leg_open_composite_quantized_then_gaussian(
            NA, t1, t2, wallA, Ms_A,
            k_quant, p_round, sigma,
            mode=comp_mode, weights=weights_A, rng=rng
        )
        HB, vB, eB = LCH.build_single_leg_open_composite_quantized_then_gaussian(
            NB, t1, t2, wallB, Ms_B,
            k_quant, p_round, sigma,
            mode=comp_mode, weights=weights_B, rng=rng
        )
        uA, _ = LCH.extract_wall_qubit_frame(eA, NA, wallA, which_block="A")
        uB, _ = LCH.extract_wall_qubit_frame(eB, NB, wallB, which_block="A")
        WA, WB = CHEX.frame_unitary_from_basis(uA), CHEX.frame_unitary_from_basis(uB)
        Svals.append(CHEX.S_fixed_in_frames(psi, a, ap, b, bp, WA, WB))
    return float(np.mean(Svals))

# ----------------------------
# Scans
# ----------------------------

def scan_tau_1D(
    taus: np.ndarray,
    *,  # fixed params
    NA, NB, Ms_A, Ms_B, t1, t2, k_quant, p_round, sigma, seeds,
    comp_mode, weights_A, weights_B,
    angles=(0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi),
    Jprep=1.0
):
    S_list = []
    for tau in taus:
        psi = prep_bell_like_state(float(tau), Jprep=Jprep)
        S = S_fixed_for_legs(
            NA, NB, Ms_A, Ms_B, t1, t2, k_quant, p_round, sigma, seeds,
            psi, angles, comp_mode=comp_mode, weights_A=weights_A, weights_B=weights_B
        )
        S_list.append(S)
    return np.array(S_list)

def heatmap_k_tau(
    k_vals: np.ndarray,
    tau_vals: np.ndarray,
    *,  # fixed params
    NA, NB, Ms_A, Ms_B, t1, t2, p_round, sigma, seeds,
    comp_mode, weights_A, weights_B,
    angles=(0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi),
    Jprep=1.0
):
    Z = np.zeros((len(tau_vals), len(k_vals)))
    for it, tau in enumerate(tau_vals):
        psi = prep_bell_like_state(float(tau), Jprep=Jprep)
        for ik, kq in enumerate(k_vals):
            S = S_fixed_for_legs(
                NA, NB, Ms_A, Ms_B, t1, t2, int(kq), p_round, sigma, seeds,
                psi, angles, comp_mode=comp_mode, weights_A=weights_A, weights_B=weights_B
            )
            Z[it, ik] = S
    return Z

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase-locked entanglers: τ scans and (k, τ) heatmaps")
    # Lattice
    ap.add_argument("--NA", type=int, default=41)
    ap.add_argument("--NB", type=int, default=41)
    ap.add_argument("--t1", type=float, default=0.7)
    ap.add_argument("--t2", type=float, default=1.3)

    # Composite moduli
    ap.add_argument("--Ms_A", type=str, default="3,5", help="comma list, e.g. 3,5")
    ap.add_argument("--Ms_B", type=str, default="9",   help="comma list, e.g. 9")
    ap.add_argument("--comp_mode", choices=["lcm","sum"], default="lcm")

    # Decoherence / quantization
    ap.add_argument("--k", type=int, default=18, help="phase quantization k (for 1D τ scan)")
    ap.add_argument("--p_round", type=float, default=0.9)
    ap.add_argument("--sigma", type=float, default=0.20)

    # Tau ranges
    ap.add_argument("--tau_min", type=float, default=0.0)
    ap.add_argument("--tau_max", type=float, default=np.pi/2)  # will include π/9, π/15, etc.
    ap.add_argument("--tau_pts", type=int, default=25)

    # Heatmap over k × τ
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=45)
    ap.add_argument("--kstep", type=int, default=1)

    # State prep
    ap.add_argument("--Jprep", type=float, default=1.0)

    # Averaging
    ap.add_argument("--seeds", type=int, default=16)

    # Modes / output
    ap.add_argument("--mode", choices=["tau", "heatmap"], default="tau")
    ap.add_argument("--save", type=str, default=None)

    args = ap.parse_args()

    Ms_A = [int(x) for x in args.Ms_A.split(",") if x.strip()]
    Ms_B = [int(x) for x in args.Ms_B.split(",") if x.strip()]

    # Angles: Tsirelson-optimal for |Φ+>
    angles = (0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi)

    if args.mode == "tau":
        taus = np.linspace(args.tau_min, args.tau_max, args.tau_pts)
        S_tau = scan_tau_1D(
            taus,
            NA=args.NA, NB=args.NB, Ms_A=Ms_A, Ms_B=Ms_B,
            t1=args.t1, t2=args.t2,
            k_quant=args.k, p_round=args.p_round, sigma=args.sigma, seeds=args.seeds,
            comp_mode=args.comp_mode, weights_A=None, weights_B=None,
            angles=angles, Jprep=args.Jprep
        )

        # Plot
        plt.figure(figsize=(9,4.5))
        plt.plot(taus, S_tau, marker='o')
        # mark special locks
        locks = []
        # τ = π/9, π/15, π/18 (LCM lines for {9, 15})
        locks += [np.pi/9, np.pi/15, np.pi/18]
        # Also π/3, π/5 for visibility
        locks += [np.pi/3, np.pi/5]
        for tlock in locks:
            if args.tau_min <= tlock <= args.tau_max:
                plt.axvline(tlock, ls='--', alpha=0.4, color='gray')
                plt.text(tlock, np.min(S_tau)-0.05, f"π/{int(round(np.pi/tlock))}", rotation=90,
                         va='bottom', ha='center', fontsize=8, alpha=0.7)
        plt.axhline(2.0,     ls='--', label='Classical bound')
        plt.axhline(2.8284,  ls=':',  label='Tsirelson')
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$\langle S_{\rm fixed}\rangle$")
        plt.title(f"Phase-locked entangler scan | Ms_A={Ms_A}, Ms_B={Ms_B}, k={args.k}, p_round={args.p_round}, σ={args.sigma}")
        plt.legend()
        plt.tight_layout()
        if args.save: plt.savefig(args.save, dpi=160)
        else:         plt.show()

    else:  # heatmap
        k_vals   = np.arange(args.kmin, args.kmax+1, args.kstep)
        tau_vals = np.linspace(args.tau_min, args.tau_max, args.tau_pts)
        Z = heatmap_k_tau(
            k_vals, tau_vals,
            NA=args.NA, NB=args.NB, Ms_A=Ms_A, Ms_B=Ms_B,
            t1=args.t1, t2=args.t2,
            p_round=args.p_round, sigma=args.sigma, seeds=args.seeds,
            comp_mode=args.comp_mode, weights_A=None, weights_B=None,
            angles=angles, Jprep=args.Jprep
        )

        plt.figure(figsize=(12,5))
        im = plt.imshow(Z, aspect="auto", origin="lower",
                        extent=[k_vals[0], k_vals[-1], tau_vals[0], tau_vals[-1]])
        plt.colorbar(im, label=r"$\langle S_{\rm fixed}\rangle$")
        # mark lock lines
        for tlock in [np.pi/9, np.pi/15, np.pi/18, np.pi/3, np.pi/5]:
            if args.tau_min <= tlock <= args.tau_max:
                plt.axhline(tlock, ls='--', color='white', alpha=0.5)
        plt.axhline(y=0, color='k', lw=0.5)
        plt.axvline(x=0, color='k', lw=0.5)
        plt.xlabel("Phase quantization k")
        plt.ylabel(r"$\tau$")
        plt.title(f"Heatmap: ⟨S_fixed⟩ vs (k, τ) | Ms_A={Ms_A}, Ms_B={Ms_B}, p_round={args.p_round}, σ={args.sigma}, mode={args.comp_mode}")
        plt.tight_layout()
        if args.save: plt.savefig(args.save, dpi=160)
        else:         plt.show()

if __name__ == "__main__":
    main()
