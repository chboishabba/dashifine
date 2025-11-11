# ============================================
# runner_phase_offset_scan.py
# --------------------------------------------
# τ × δ scan of lattice-constrained CHSH with a relative
# phase offset δ injected into the local measurement frames.
#
# Requirements:
#   - lattice_chsh.py must expose:
#       * build_single_leg_open_modulus_quantized_random(...)
#       * extract_wall_qubit_frame(...)
#   - chsh_extras.py must expose:
#       * pauli(), kron2(), frame_unitary_from_basis(...),
#         S_fixed_in_frames(...), ket_to_rho(...), depolarize_rho(...),
#         (optional) any helpers you already use elsewhere
#
# Example:
#   python runner_phase_offset_scan.py \
#       --NA 41 --NB 41 --M_A 6 --M_B 9 --k 18 \
#       --p_round 0.9 --seeds 32 \
#       --tau_min 0.0 --tau_max 1.8 --tau_steps 61 \
#       --delta_steps 121 --out_prefix tau_delta_scan
#
# Optional FFT of a single τ-row:
#   python runner_phase_offset_scan.py ... --fft_row 30
#
# Notes
#   - We rotate the *local* Bloch frames by ±δ about Z:
#       W_A(δ) = W_A @ Rz(+δ),  W_B(δ) = W_B @ Rz(-δ)
#     so δ is a *relative* phase between legs.
#   - We keep Tsirelson measurement angles (a=0, a'=π/2, b=π/4, b'=-π/4)
#     inside those (rotated) frames; this intentionally avoids global
#     Horodecki optimization to keep sensitivity to the 3–6–9 substrate.
# ============================================

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

import lattice_chsh as LCH
import chsh_extras as CHEX

# -----------------------------
# SU(2) Z-rotation (Bloch Z-axis)
# -----------------------------
def Rz(theta: float) -> np.ndarray:
    # e^{-i θ σ_z / 2} = diag(e^{-iθ/2}, e^{+iθ/2})
    half = 0.5 * theta
    return np.array([[np.exp(-1j * half), 0.0],
                     [0.0,                np.exp(+1j * half)]], dtype=complex)

# -----------------------------
# Build one leg and extract its wall qubit frame
# -----------------------------
def build_leg_frame_random(N: int, t1: float, t2: float, wall: int,
                           M: int, k_quant: int, p_round: float,
                           seed: int):
    rng = np.random.default_rng(seed)
    H, vals, vecs = LCH.build_single_leg_open_modulus_quantized_random(
        N, t1, t2, wall, M, k_quant, p_round=p_round, rng=rng
    )
    # Use A-block wall mode as you’ve done elsewhere
    u, _ = LCH.extract_wall_qubit_frame(vecs, N, wall, which_block="A")
    W = CHEX.frame_unitary_from_basis(u)
    return W

# -----------------------------
# Prepare entangled state via Heisenberg-like entangler
# -----------------------------
def prepare_state(tau: float, Jprep: float = 1.0) -> np.ndarray:
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx, sx) + CHEX.kron2(sy, sy) + CHEX.kron2(sz, sz)
    psi0 = np.zeros(4, complex); psi0[1] = 1.0   # |01> seed (your convention)
    U = expm(-1j * Jprep * tau * H2)
    return U @ psi0

# -----------------------------
# Compute S_fixed with lattice-constrained frames and a relative δ
# -----------------------------
def S_fixed_with_delta(psi: np.ndarray,
                       WA: np.ndarray, WB: np.ndarray,
                       delta: float) -> float:
    # Tsirelson angles in local frame
    a, ap, b, bp = 0.0, 0.5 * np.pi, 0.25 * np.pi, -0.25 * np.pi
    WA_d = WA @ Rz(+delta)
    WB_d = WB @ Rz(-delta)
    return CHEX.S_fixed_in_frames(psi, a, ap, b, bp, WA_d, WB_d)

# -----------------------------
# Grid scan over τ and δ (seed-averaged)
# -----------------------------
def scan_tau_delta(NA: int, NB: int, M_A: int, M_B: int, k_quant: int,
                   p_round: float, seeds: int,
                   t1: float, t2: float,
                   tau_vals: np.ndarray, delta_vals: np.ndarray):
    wallA, wallB = NA // 2, NB // 2
    S_mean = np.zeros((len(tau_vals), len(delta_vals)), dtype=float)

    for si, tau in enumerate(tau_vals):
        psi = prepare_state(tau, Jprep=1.0)
        # Build fresh frames each τ (you can lift out if you prefer “static disorder”)
        WA_stack, WB_stack = [], []
        for s in range(seeds):
            WA = build_leg_frame_random(NA, t1, t2, wallA, M_A, k_quant, p_round, seed=1000 + s)
            WB = build_leg_frame_random(NB, t1, t2, wallB, M_B, k_quant, p_round, seed=2000 + s)
            WA_stack.append(WA); WB_stack.append(WB)

        for dj, delta in enumerate(delta_vals):
            acc = 0.0
            for s in range(seeds):
                acc += S_fixed_with_delta(psi, WA_stack[s], WB_stack[s], delta)
            S_mean[si, dj] = acc / float(seeds)

    return S_mean

# -----------------------------
# Optional 1D FFT over δ at a chosen τ-row
# -----------------------------
def fft_over_delta(S_mean: np.ndarray, delta_vals: np.ndarray, row_index: int):
    # Simple DFT magnitude vs cycles per Δ-step
    y = S_mean[row_index, :]
    Y = np.fft.rfft(y - y.mean())
    freqs = np.fft.rfftfreq(len(y), d=1.0)  # “cycles per index”
    power = np.abs(Y) ** 2
    return freqs, power

# -----------------------------
# Plotting helpers
# -----------------------------
def plot_heatmap(S_mean: np.ndarray, tau_vals: np.ndarray, delta_vals: np.ndarray,
                 title: str, out_png: str):
    plt.figure(figsize=(10, 5.2), dpi=140)
    extent = [delta_vals[0], delta_vals[-1], tau_vals[0], tau_vals[-1]]
    im = plt.imshow(S_mean, origin='lower', aspect='auto', cmap='viridis',
                    extent=extent, interpolation='nearest')
    plt.colorbar(im, label=r'$\langle S_{\mathrm{fixed}} \rangle$')
    plt.xlabel(r'$\delta$ (radians)')
    plt.ylabel(r'$\tau$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[saved] {out_png}")

def plot_fft(freqs, power, row_tau, out_png: str):
    plt.figure(figsize=(9, 3.6), dpi=140)
    plt.plot(freqs, power, marker='o', lw=1.2)
    plt.xlabel('cycles per δ-step')
    plt.ylabel('power')
    plt.title(f'DFT over δ at τ ≈ {row_tau:.3f}')
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[saved] {out_png}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="τ × δ phase-offset scan with lattice-constrained CHSH")
    ap.add_argument("--NA", type=int, default=41)
    ap.add_argument("--NB", type=int, default=41)
    ap.add_argument("--M_A", type=int, default=6)
    ap.add_argument("--M_B", type=int, default=9)
    ap.add_argument("--k", type=int, default=18, help="phase quantization (k) used in rounding")
    ap.add_argument("--p_round", type=float, default=0.9, help="probability of rounding each phase")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--t1", type=float, default=0.7)
    ap.add_argument("--t2", type=float, default=1.3)

    ap.add_argument("--tau_min", type=float, default=0.0)
    ap.add_argument("--tau_max", type=float, default=np.pi/1.7)
    ap.add_argument("--tau_steps", type=int, default=61)

    ap.add_argument("--delta_min", type=float, default=-np.pi)
    ap.add_argument("--delta_max", type=float, default=+np.pi)
    ap.add_argument("--delta_steps", type=int, default=121)

    ap.add_argument("--fft_row", type=int, default=-1, help="row index for δ-FFT (optional)")
    ap.add_argument("--out_prefix", type=str, default="tau_delta_scan")

    args = ap.parse_args()

    tau_vals   = np.linspace(args.tau_min,   args.tau_max,   args.tau_steps)
    delta_vals = np.linspace(args.delta_min, args.delta_max, args.delta_steps)

    S_mean = scan_tau_delta(
        NA=args.NA, NB=args.NB, M_A=args.M_A, M_B=args.M_B, k_quant=args.k,
        p_round=args.p_round, seeds=args.seeds,
        t1=args.t1, t2=args.t2,
        tau_vals=tau_vals, delta_vals=delta_vals
    )

    title = (r"Heatmap: $\langle S_{\mathrm{fixed}}\rangle$ vs $(\delta,\tau)$ "
             f"| $M_A={args.M_A}$, $M_B={args.M_B}$, k={args.k}, p_round={args.p_round}, seeds={args.seeds}")
    plot_heatmap(S_mean, tau_vals, delta_vals, title, f"{args.out_prefix}_heatmap.png")

    if args.fft_row >= 0 and args.fft_row < len(tau_vals):
        freqs, power = fft_over_delta(S_mean, delta_vals, args.fft_row)
        plot_fft(freqs, power, tau_vals[args.fft_row], f"{args.out_prefix}_fft_row{args.fft_row}.png")
    np.savez(
        "tau_delta_scan_heatmap.npz",
        S_mean=S_mean,
        tau_vals=tau_vals,
        delta_vals=delta_vals,
    )

if __name__ == "__main__":
    main()
