# analyze_tau_delta_coupling.py
# ------------------------------------------------------------
# Quantifies τ–δ coupling in S_mean(τ, δ) by:
#   1) FFT over δ for each τ-row  → dominant δ-harmonics vs τ
#   2) FFT over τ for each δ-col  → reciprocal harmonics vs δ
#   3) Ridge-slope map ω(τ, δ) ≈ - (∂S/∂τ) / (∂S/∂δ)
#
# INPUTS (choose ONE of the following):
#   A) --npz path/to/scan.npz  (expects keys: S_mean, tau_vals, delta_vals)
#   B) --S pathS.npy --tau pathTau.npy --delta pathDelta.npy
#
# OPTIONAL:
#   --outdir DIR      (default: ./coupling_analysis)
#   --logpower        (plot log10 power instead of linear)
#   --fft_smooth N    (boxcar smooth in frequency bins; default 1 = none)
#   --ridge_quantile q  (0<q<1, gradient magnitude quantile for masking; default 0.80)
#   --savefig         (write PNG files to outdir)
#
# OUTPUTS:
#   - fft_delta_power.png      (τ × harmonic-index power map, δ-FFT)
#   - fft_tau_power.png        (δ × harmonic-index power map, τ-FFT)
#   - omega_slope_map.png      (ω(τ,δ) from gradients; also prints robust stats)
#   - omega_hist.png           (histogram of ω samples under ridge mask)
#   - coupling_metrics.txt     (median/percentiles of ω and top harmonic indices)
#   - analysis.npz             (all derived arrays for reuse)
#
# Units/notes:
#   * delta_vals are in radians (typically [-π, π]); τ is unitless “time”.
#   * δ-FFT frequency is in cycles per radian; we also report “harmonic index”
#     h = 2π * f (≈ number of bright bands around one full 2π in δ).
#   * For S(τ,δ) ~ cos(δ - ω τ + φ), the ridge slope is dδ/dτ = ω.
# ------------------------------------------------------------

from __future__ import annotations
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def load_arrays(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.npz:
        z = np.load(args.npz)
        S = z["S_mean"]; tau = z["tau_vals"]; delta = z["delta_vals"]
        return S, tau, delta
    if args.S and args.tau and args.delta:
        return np.load(args.S), np.load(args.tau), np.load(args.delta)
    raise SystemExit("Provide either --npz scan.npz OR all of --S --tau --delta.")

def _smooth_boxcar(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1: return x
    k = n
    pad = (k-1)//2
    xpad = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k)/k
    return np.convolve(xpad, ker, mode="valid")

def fft_over_delta(S: np.ndarray, delta_vals: np.ndarray, fft_smooth: int = 1):
    """
    Returns:
      harmonics_h  (length ~ Nδ//2+1):  h = 2π * fδ  (dimensionless harmonic index)
      power_map    (Nτ, Nh): power vs τ × harmonic
    """
    dδ = float(delta_vals[1] - delta_vals[0])
    freqs = np.fft.rfftfreq(delta_vals.size, d=dδ)      # cycles per radian
    h = 2*np.pi*freqs                                   # “harmonic index”
    Nτ, Nδ = S.shape
    P = np.empty((Nτ, freqs.size), float)
    for i in range(Nτ):
        row = S[i, :] - S[i, :].mean()
        Y = np.fft.rfft(row)
        pow_ = np.abs(Y)**2
        if fft_smooth > 1:
            pow_ = _smooth_boxcar(pow_, fft_smooth)
        P[i, :] = pow_
    return h, P

def fft_over_tau(S: np.ndarray, tau_vals: np.ndarray, fft_smooth: int = 1):
    """
    Returns:
      freq_tau     (cycles per τ-unit)
      power_map    (Nδ, Nfτ): power vs δ × fτ
    """
    dτ = float(tau_vals[1] - tau_vals[0])
    freqsτ = np.fft.rfftfreq(tau_vals.size, d=dτ)       # cycles per τ-unit
    Nτ, Nδ = S.shape
    P = np.empty((Nδ, freqsτ.size), float)
    for j in range(Nδ):
        col = S[:, j] - S[:, j].mean()
        Y = np.fft.rfft(col)
        pow_ = np.abs(Y)**2
        if fft_smooth > 1:
            pow_ = _smooth_boxcar(pow_, fft_smooth)
        P[j, :] = pow_
    return freqsτ, P

def slope_map_omega(S: np.ndarray, tau_vals: np.ndarray, delta_vals: np.ndarray, ridge_quantile: float = 0.80):
    """
    ω(τ,δ) ≈ - (∂S/∂τ) / (∂S/∂δ) under a ridge mask based on gradient magnitude.
    Returns:
      omega_map        (Nτ, Nδ) with NaNs off-ridge
      ridge_mask       (bool map)
      stats            (dict with robust ω summary)
    """
    dS_dτ, dS_dδ = np.gradient(S, tau_vals, delta_vals, edge_order=2)
    grad_mag = np.hypot(dS_dτ, dS_dδ)
    thr = np.quantile(grad_mag, ridge_quantile)
    ridge = grad_mag >= thr

    # Avoid division issues
    eps = 1e-12
    denom = np.where(np.abs(dS_dδ) < eps, np.nan, dS_dδ)
    omega = -dS_dτ / denom
    omega_map = np.where(ridge, omega, np.nan)

    # Robust stats
    samples = omega_map[np.isfinite(omega_map)]
    stats = {
        "count": int(samples.size),
        "median": float(np.nanmedian(samples)) if samples.size else np.nan,
        "p10": float(np.nanpercentile(samples, 10)) if samples.size else np.nan,
        "p90": float(np.nanpercentile(samples, 90)) if samples.size else np.nan,
        "mean": float(np.nanmean(samples)) if samples.size else np.nan,
        "std": float(np.nanstd(samples)) if samples.size else np.nan,
        "quantile_threshold": float(thr),
    }
    return omega_map, ridge, stats

def main():
    ap = argparse.ArgumentParser(description="Analyze τ–δ coupling in S_mean(τ, δ).")
    ap.add_argument("--npz", type=str, help="npz containing S_mean, tau_vals, delta_vals")
    ap.add_argument("--S", type=str, help="npy path for S_mean")
    ap.add_argument("--tau", type=str, help="npy path for tau_vals")
    ap.add_argument("--delta", type=str, help="npy path for delta_vals")
    ap.add_argument("--outdir", type=str, default="coupling_analysis")
    ap.add_argument("--logpower", action="store_true", help="plot log10 power in FFT maps")
    ap.add_argument("--fft_smooth", type=int, default=1)
    ap.add_argument("--ridge_quantile", type=float, default=0.80)
    ap.add_argument("--savefig", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    S, tau_vals, delta_vals = load_arrays(args)
    # Ensure shape is (Nτ, Nδ)
    if S.shape[0] != tau_vals.size or S.shape[1] != delta_vals.size:
        raise SystemExit(f"Shape mismatch: S{S.shape}, tau({tau_vals.size}), delta({delta_vals.size})")

    # ---------- FFT over δ ----------
    h, Pδ = fft_over_delta(S, delta_vals, fft_smooth=args.fft_smooth)
    Pδ_plot = np.log10(Pδ + 1e-12) if args.logpower else Pδ
    plt.figure(figsize=(9, 5))
    im = plt.imshow(Pδ_plot, aspect="auto", origin="lower",
                    extent=[h[0], h[-1], tau_vals[0], tau_vals[-1]], cmap="viridis")
    plt.colorbar(im, label="log10 power" if args.logpower else "power")
    plt.xlabel("δ-harmonic index  h = 2π fδ")
    plt.ylabel("τ")
    plt.title("FFT over δ  (power vs τ × harmonic)")
    if args.savefig:
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "fft_delta_power.png"), dpi=180)
    else:
        plt.show()

    # Also estimate the top harmonic index per τ (for report)
    top_h_idx = np.argmax(Pδ, axis=1)  # per-row
    top_h = h[top_h_idx]

    # ---------- FFT over τ ----------
    fτ, Pτ = fft_over_tau(S, tau_vals, fft_smooth=args.fft_smooth)
    Pτ_plot = np.log10(Pτ + 1e-12) if args.logpower else Pτ
    plt.figure(figsize=(9, 5))
    im = plt.imshow(Pτ_plot.T, aspect="auto", origin="lower",
                    extent=[delta_vals[0], delta_vals[-1], fτ[0], fτ[-1]], cmap="viridis")
    plt.colorbar(im, label="log10 power" if args.logpower else "power")
    plt.xlabel("δ (radians)")
    plt.ylabel("τ-FFT frequency (cycles per τ-unit)")
    plt.title("FFT over τ  (power vs δ × fτ)")
    if args.savefig:
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "fft_tau_power.png"), dpi=180)
    else:
        plt.show()

    # ---------- Ridge slope map ω(τ, δ) ----------
    omega_map, ridge_mask, stats = slope_map_omega(
        S, tau_vals, delta_vals, ridge_quantile=args.ridge_quantile
    )

    plt.figure(figsize=(9, 5))
    im = plt.imshow(omega_map, aspect="auto", origin="lower",
                    extent=[delta_vals[0], delta_vals[-1], tau_vals[0], tau_vals[-1]],
                    cmap="coolwarm")
    plt.colorbar(im, label="ω ≈ - (∂S/∂τ)/(∂S/∂δ)")
    plt.contour(delta_vals, tau_vals, ridge_mask, levels=[0.5], colors="k", linewidths=0.8)
    plt.xlabel("δ (radians)")
    plt.ylabel("τ")
    plt.title(f"Ridge-slope ω-map  |  median={stats['median']:.3f}, p10={stats['p10']:.3f}, p90={stats['p90']:.3f}")
    if args.savefig:
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "omega_slope_map.png"), dpi=180)
    else:
        plt.show()

    # Histogram of ω samples
    samples = omega_map[np.isfinite(omega_map)]
    plt.figure(figsize=(7,4))
    plt.hist(samples, bins=60, density=True, alpha=0.85)
    plt.axvline(stats["median"], color="k", ls="--", label=f"median={stats['median']:.3f}")
    plt.xlabel("ω")
    plt.ylabel("density")
    plt.title("Distribution of ω from ridge slopes")
    plt.legend()
    if args.savefig:
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "omega_hist.png"), dpi=180)
    else:
        plt.show()

    # ---------- Save artifacts ----------
    np.savez_compressed(
        os.path.join(args.outdir, "analysis.npz"),
        S_mean=S, tau_vals=tau_vals, delta_vals=delta_vals,
        h=h, P_delta=Pδ, f_tau=fτ, P_tau=Pτ,
        omega_map=omega_map, ridge_mask=ridge_mask, top_h=top_h
    )
    with open(os.path.join(args.outdir, "coupling_metrics.txt"), "w") as f:
        f.write("# Coupling metrics\n")
        f.write(json.dumps({
            "ridge_stats": stats,
            "top_harmonic_index_summary": {
                "median_h": float(np.median(top_h)),
                "mean_h": float(np.mean(top_h)),
                "p10_h": float(np.percentile(top_h, 10)),
                "p90_h": float(np.percentile(top_h, 90))
            }
        }, indent=2))
    print("\n=== SUMMARY ===")
    print(f"ω samples: count={stats['count']}  median={stats['median']:.4f}  "
          f"p10={stats['p10']:.4f}  p90={stats['p90']:.4f}  mean={stats['mean']:.4f}  std={stats['std']:.4f}")
    print(f"Top δ-harmonic index h (per-τ) → median={np.median(top_h):.3f}, mean={np.mean(top_h):.3f}")
    print(f"Outputs written to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
