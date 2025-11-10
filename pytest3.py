#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from math import floor

def build_signals(N: int, mmax: int):
    """
    Construct sequences on x = 0..N-1 for moduli m = 1..mmax:
      constructive(x) = sum_m (x mod m)
      destructive(x)  = sum_m min(r, m-r) / (m/2)   where r = x mod m
    Returns (x, constructive, destructive)
    """
    x = np.arange(N, dtype=np.int64)
    constructive = np.zeros(N, dtype=np.float64)
    destructive  = np.zeros(N, dtype=np.float64)

    for m in range(1, mmax + 1):
        r = x % m
        constructive += r
        # circular distance to nearest multiple in [0, m/2]
        dist = np.minimum(r, (m - r) % m).astype(np.float64)
        destructive += dist / (m / 2.0)

    return x, constructive, destructive

def rfft_spectrum(signal: np.ndarray):
    """
    Hann-windowed rFFT with amplitude normalization.
    Returns (freqs, amps) where freqs are in cycles/sample.
    """
    N = signal.size
    s = signal - signal.mean()
    window = np.hanning(N)
    s_win = s * window
    fft_vals = np.fft.rfft(s_win)
    freqs = np.fft.rfftfreq(N, d=1.0)
    # scale by window sum for amplitude-like units
    amps = np.abs(fft_vals) * 2.0 / window.sum()
    return freqs, amps

def simple_peaks(freqs, amps, min_sep=5, k_top=20, fmin=0.0):
    """
    Minimal peak picker: local maxima with optional frequency floor and
    minimum index separation, then take top-k by amplitude.
    """
    mask = freqs >= fmin
    f = freqs[mask]
    a = amps[mask]
    # local maxima
    peaks = []
    for i in range(1, len(a) - 1):
        if a[i] > a[i - 1] and a[i] > a[i + 1]:
            peaks.append((i, a[i]))
    # sort by amplitude desc
    peaks.sort(key=lambda t: t[1], reverse=True)
    # enforce min_sep in index space
    selected = []
    used = np.zeros(len(a), dtype=bool)
    for idx, _amp in peaks:
        if used[max(0, idx - min_sep):min(len(a), idx + min_sep + 1)].any():
            continue
        selected.append(idx)
        used[max(0, idx - min_sep):min(len(a), idx + min_sep + 1)] = True
        if len(selected) >= k_top:
            break
    return f[selected], a[selected]

def reference_lines(mmax: int, nyquist: float = 0.5):
    """
    All rational lines f = k/m with 1<=m<=mmax and 1<=k such that f <= nyquist.
    Returns sorted unique frequencies.
    """
    refs = set()
    for m in range(1, mmax + 1):
        k = 1
        while True:
            f = k / m
            if f > nyquist + 1e-12:
                break
            refs.add(round(f, 10))  # dedupe numerically
            k += 1
    return sorted(refs)

def save_csv(path, freqs, amp_c, amp_d_neg):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frequency_cycles_per_sample",
                    "constructive_amplitude",
                    "negative_destructive_amplitude"])
        for fr, ac, ad in zip(freqs, amp_c, amp_d_neg):
            w.writerow([f"{fr:.12f}", f"{ac:.12e}", f"{ad:.12e}"])

def plot_overlay(freqs, amp_c, amp_d_neg, mmax, out_png):
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, amp_c, label="Constructive spectrum (Σ residues)")
    plt.plot(freqs, amp_d_neg, label="− Destructive spectrum (distance sum)")
    # reference harmonics
    for f in reference_lines(mmax):
        plt.axvline(f, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.title(f"Overlaid Spectra (N={len(freqs)*2-2} samples): Constructive vs −Destructive")
    plt.xlabel("Frequency (cycles/sample)")
    plt.ylabel("Amplitude (constructive up / destructive down)")
    plt.xlim(0, 0.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=180)
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Modulo interference FFT (constructive vs −destructive)")
    ap.add_argument("--N", type=int, default=100000, help="number of samples (x = 0..N-1)")
    ap.add_argument("--mmax", type=int, default=11, help="max modulus (use 11 for 1..11)")
    ap.add_argument("--peaks", type=int, default=0, help="print top-K peaks (0 to skip)")
    ap.add_argument("--csv", default="fft_constructive_vs_negative_destructive.csv", help="output CSV path")
    ap.add_argument("--png", default="fft_overlay.png", help="output PNG path")
    args = ap.parse_args()

    # 1) Build sequences
    x, constructive, destructive = build_signals(args.N, args.mmax)

    # 2) FFTs
    freq_c, amp_c = rfft_spectrum(constructive)
    freq_d, amp_d = rfft_spectrum(destructive)
    if not np.allclose(freq_c, freq_d):
        raise RuntimeError("Frequency grids differ (unexpected).")
    freq = freq_c
    amp_d_neg = -amp_d  # invert for below-axis plotting

    # 3) Save + plot
    save_csv(args.csv, freq, amp_c, amp_d_neg)
    plot_overlay(freq, amp_c, amp_d_neg, args.mmax, args.png)

    # 4) Optional peaks
    if args.peaks > 0:
        fpk_c, apk_c = simple_peaks(freq, amp_c, k_top=args.peaks, fmin=1.0/args.N)
        fpk_d, apk_d = simple_peaks(freq, amp_d, k_top=args.peaks, fmin=1.0/args.N)
        print("\nTop constructive peaks (freq, period, amp):")
        for f, a in zip(fpk_c, apk_c):
            print(f"  {f:.6f}  (period ~ {1.0/f:.2f})   {a:.6f}")
        print("\nTop destructive peaks (freq, period, amp):")
        for f, a in zip(fpk_d, apk_d):
            print(f"  {f:.6f}  (period ~ {1.0/f:.2f})   {a:.6f}")

if __name__ == "__main__":
    main()
