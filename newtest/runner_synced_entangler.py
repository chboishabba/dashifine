# ============================================
# runner_synced_entangler.py
# Phase-locked entangler scan over (tau, delta)
# Defaults: M_A=6, M_B=9, k=18, p_round=0.9, seeds=32
# Saves an NPZ you can feed to analyze_tau_delta_coupling.py
# ============================================

from __future__ import annotations
import argparse
import numpy as np
from scipy.linalg import expm

import lattice_chsh as LCH
import chsh_extras as CHEX

# ---------- Pauli / kron helpers ----------
sx, sy, sz = CHEX.pauli()
XZ = CHEX.kron2(sx, sz)
ZX = CHEX.kron2(sz, sx)

def H2_synced(tau: float, omega: float, delta: float = 0.0) -> np.ndarray:
    """
    Synchronized (relative-phase) Heisenberg-like entangler.
      H2(τ) = cos(ωτ + δ) (σx ⊗ σz) + sin(ωτ + δ) (σz ⊗ σx)
    """
    ph = omega * float(tau) + float(delta)
    return np.cos(ph) * XZ + np.sin(ph) * ZX

def build_leg_random_quantized(N: int, t1: float, t2: float,
                               wall: int, M: int, k_quant: int,
                               p_round: float, seed: int):
    rng = np.random.default_rng(seed)
    H, vals, vecs = LCH.build_single_leg_open_modulus_quantized_random(
        N, t1, t2, wall, M, k_quant, p_round=p_round, rng=rng
    )
    # measurement frame from domain-wall A-block
    uA, _ = LCH.extract_wall_qubit_frame(vecs, N, wall, which_block="A")
    return H, vals, vecs, uA

def compute_S_fixed_for_frames(psi, a, ap, b, bp, WA, WB):
    return CHEX.S_fixed_in_frames(psi, a, ap, b, bp, WA, WB)

def main():
    p = argparse.ArgumentParser(description="Phase-locked entangler scan (tau, delta)")
    # Lattice / modulus
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--M_A", type=int, default=6)
    p.add_argument("--M_B", type=int, default=9)
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    p.add_argument("--k", type=int, default=18, help="phase quantization for rounding")
    p.add_argument("--p_round", type=float, default=0.9)
    p.add_argument("--seeds", type=int, default=32)

    # Entangler
    p.add_argument("--omega_lock", type=float, default=0.483,
                   help="measured lock frequency ω (rad per τ-unit)")
    p.add_argument("--Jprep", type=float, default=1.0,
                   help="global scale in exp(-i * Jprep * τ * H2(τ))")

    # Scan ranges
    p.add_argument("--tau_min", type=float, default=0.0)
    p.add_argument("--tau_max", type=float, default=1.8)
    p.add_argument("--tau_steps", type=int, default=72)
    p.add_argument("--delta_min", type=float, default=-np.pi)
    p.add_argument("--delta_max", type=float, default= np.pi)
    p.add_argument("--delta_steps", type=int, default=121)

    # I/O
    p.add_argument("--outfile", type=str, default="synced_scan.npz")
    p.add_argument("--plot", action="store_true")

    args = p.parse_args()

    wallA, wallB = args.NA // 2, args.NB // 2

    # Prepare singlet-like initial state (same as your runners)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0  # |01>
    # Use same CHSH angles you’ve been using (Tsirelson set in local frames)
    a, ap, b, bp = 0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi

    tau_vals   = np.linspace(args.tau_min,   args.tau_max,   args.tau_steps)
    delta_vals = np.linspace(args.delta_min, args.delta_max, args.delta_steps)

    # Allocate
    S_mean = np.zeros((args.tau_steps, args.delta_steps), dtype=float)

    print("Scanning (τ, δ) with synchronized entangler...")
    print(f"  ω_lock = {args.omega_lock:.3f}, Jprep = {args.Jprep}")
    print(f"  M_A={args.M_A}, M_B={args.M_B}, k={args.k}, p_round={args.p_round}, seeds={args.seeds}")
    print(f"  τ ∈ [{args.tau_min}, {args.tau_max}] x {args.tau_steps} | "
          f"δ ∈ [{args.delta_min}, {args.delta_max}] x {args.delta_steps}")

    for it, tau in enumerate(tau_vals):
        S_row = np.zeros(args.delta_steps, dtype=float)

        # Build fresh frames each τ-row (keeps coupling to lattice “live”)
        Svals_accum = np.zeros(args.delta_steps, dtype=float)
        for seed in range(args.seeds):
            # Two legs with rounding noise (granular decoherence)
            _, _, _, uA = build_leg_random_quantized(
                args.NA, args.t1, args.t2, wallA, args.M_A, args.k, args.p_round, seed=seed
            )
            _, _, _, uB = build_leg_random_quantized(
                args.NB, args.t1, args.t2, wallB, args.M_B, args.k, args.p_round, seed=10_000+seed
            )
            WA = CHEX.frame_unitary_from_basis(uA)
            WB = CHEX.frame_unitary_from_basis(uB)

            # Phase-locked entangler at this τ, with δ swept below
            S_this_seed = np.empty(args.delta_steps, dtype=float)
            for jd, delta in enumerate(delta_vals):
                H2 = H2_synced(tau, args.omega_lock, delta)
                U  = expm(-1j * args.Jprep * tau * H2)
                psi = U @ psi0
                S_this_seed[jd] = compute_S_fixed_for_frames(psi, a, ap, b, bp, WA, WB)

            Svals_accum += S_this_seed

        S_mean[it, :] = Svals_accum / float(args.seeds)

    # Save NPZ (compatible with analyze_tau_delta_coupling.py)
    np.savez_compressed(args.outfile,
                        S_mean=S_mean,
                        tau_vals=tau_vals,
                        delta_vals=delta_vals,
                        meta=dict(omega_lock=args.omega_lock,
                                  Jprep=args.Jprep,
                                  M_A=args.M_A, M_B=args.M_B,
                                  k=args.k, p_round=args.p_round,
                                  seeds=args.seeds))

    print(f"Saved: {args.outfile}")
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5.2))
        plt.imshow(S_mean, origin="lower", aspect="auto",
                   extent=[delta_vals[0], delta_vals[-1], tau_vals[0], tau_vals[-1]],
                   cmap="viridis")
        cbar = plt.colorbar()
        cbar.set_label(r'$\langle S_{\rm fixed}\rangle$')
        plt.title(rf"Phase-locked: $\langle S_{{\rm fixed}}\rangle(\delta,\tau)$  "
                  rf"$\omega_{{\rm lock}}={args.omega_lock:.3f}$,  "
                  rf"$M_A={args.M_A},M_B={args.M_B},k={args.k},p={args.p_round}$")
        plt.xlabel(r'$\delta$ (radians)')
        plt.ylabel(r'$\tau$')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
