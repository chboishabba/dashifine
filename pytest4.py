#!/usr/bin/env python3
import argparse
import math
import csv
import numpy as np

# ----------------------------
# Signal builders
# ----------------------------
def build_signals(N: int, mmax: int):
    """
    x = 0..N-1
    constructive(x) = sum_m (x mod m)
    destructive(x)  = sum_m min(r, m-r) / (m/2)   where r = x mod m
    """
    x = np.arange(N, dtype=np.int64)
    constructive = np.zeros(N, dtype=np.float64)
    destructive  = np.zeros(N, dtype=np.float64)

    for m in range(1, mmax + 1):
        r = x % m
        constructive += r
        dist = np.minimum(r, (m - r) % m).astype(np.float64)
        destructive += dist / (m / 2.0)

    return x, constructive, destructive

# ----------------------------
# FFT
# ----------------------------
def rfft_spectrum(signal: np.ndarray):
    """Hann-windowed rFFT with amplitude-like normalization."""
    N = signal.size
    s = signal - signal.mean()
    window = np.hanning(N)
    s_win = s * window
    fft_vals = np.fft.rfft(s_win)
    freqs = np.fft.rfftfreq(N, d=1.0)
    amps = np.abs(fft_vals) * 2.0 / window.sum()
    return freqs, amps

# ----------------------------
# Peak picking (simple + fast)
# ----------------------------
def simple_peaks(freqs, amps, min_sep=5, k_top=50, fmin=0.0):
    """
    Local-max peak picker with minimum index separation.
    Returns top-k peaks (freqs, amps) sorted by amplitude (desc).
    """
    mask = freqs >= fmin
    f = freqs[mask]
    a = amps[mask]
    cands = []
    for i in range(1, len(a) - 1):
        if a[i] > a[i-1] and a[i] > a[i+1]:
            cands.append((i, a[i]))
    cands.sort(key=lambda t: t[1], reverse=True)

    selected = []
    used = np.zeros(len(a), dtype=bool)
    for idx, _amp in cands:
        if used[max(0, idx - min_sep): min(len(a), idx + min_sep + 1)].any():
            continue
        selected.append(idx)
        used[max(0, idx - min_sep): min(len(a), idx + min_sep + 1)] = True
        if len(selected) >= k_top:
            break

    sel_f = f[selected]
    sel_a = a[selected]
    # sort by frequency for nicer reading
    order = np.argsort(sel_f)
    return sel_f[order], sel_a[order]

# ----------------------------
# Reference grid f = k/m
# ----------------------------
def generate_reference_dict(mmax: int, nyq: float = 0.5):
    """
    Dictionary of all rational lines f = k/m (1<=m<=mmax, k>=1) with f <= nyq.
    Maps frequency -> list of (k, m, period=m/k).
    """
    refs = {}
    for m in range(1, mmax + 1):
        k = 1
        while True:
            f = k / m
            if f > nyq + 1e-12:
                break
            period = m / k
            refs.setdefault(f, []).append((k, m, period))
            k += 1
    return refs

def match_peaks_to_refs(peak_freqs, ref_freqs, N, tol_bins=3):
    """
    Map each peak frequency to nearest rational f_ref in ref_freqs
    within absolute tolerance tol_bins/N. Returns list of dict rows.
    """
    tol = tol_bins / N
    rows = []
    ref_sorted = np.array(sorted(ref_freqs))
    for f in peak_freqs:
        i = np.searchsorted(ref_sorted, f)
        candidates = []
        if 0 < i <= len(ref_sorted) - 1:
            candidates = [ref_sorted[i-1], ref_sorted[i]]
        elif i == 0 and len(ref_sorted):
            candidates = [ref_sorted[0]]
        elif i == len(ref_sorted) and len(ref_sorted):
            candidates = [ref_sorted[-1]]

        best = None
        best_err = 1e9
        for fr in candidates:
            err = abs(f - fr)
            if err < best_err:
                best = fr
                best_err = err

        if best is None or best_err > tol:
            rows.append({
                "peak_freq": f,
                "match_freq": math.nan,
                "match_k": None,
                "match_m": None,
                "period_samples": math.nan,
                "abs_error": best_err,
                "bins_error": best_err * N,
                "ppm_error": best_err / max(f, 1e-12) * 1e6
            })
        else:
            rows.append({
                "peak_freq": f,
                "match_freq": best,
                "abs_error": best_err,
                "bins_error": best_err * N,
                "ppm_error": best_err / best * 1e6,
                "match_k": None,
                "match_m": None,
                "period_samples": None,
            })
    return rows

def enrich_with_km(rows, refs_dict):
    """
    Attach (k,m,period) for each matched frequency.
    If multiple exist (e.g., 1/2 = 2/4), prefer smallest m (then k).
    """
    for r in rows:
        fr = r.get("match_freq")
        if fr is None or (isinstance(fr, float) and math.isnan(fr)):
            continue
        key = min(refs_dict.keys(), key=lambda q: abs(q - fr))
        km_list = sorted(refs_dict[key], key=lambda t: (t[1], t[0]))
        k, m, period = km_list[0]
        r["match_k"] = int(k)
        r["match_m"] = int(m)
        r["period_samples"] = period

def save_peak_map_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "peak_freq", "match_freq", "abs_error", "bins_error", "ppm_error",
            "k", "m", "period(=m/k)"
        ])
        for r in rows:
            w.writerow([
                f"{r['peak_freq']:.12f}",
                "" if (r.get("match_freq") is None or (isinstance(r.get('match_freq'), float) and math.isnan(r['match_freq']))) else f"{r['match_freq']:.12f}",
                f"{r['abs_error']:.3e}",
                f"{r['bins_error']:.2f}",
                f"{r['ppm_error']:.2f}",
                r.get("match_k"),
                r.get("match_m"),
                "" if r.get("period_samples") is None else f"{r['period_samples']:.6f}",
            ])

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Map FFT peaks to nearest k/m (m<=mmax)")
    ap.add_argument("--N", type=int, default=100000, help="number of samples (x = 0..N-1)")
    ap.add_argument("--mmax", type=int, default=11, help="max modulus (1..mmax)")
    ap.add_argument("--peaks", type=int, default=50, help="top-K peaks to analyze per spectrum")
    args = ap.parse_args()

    # 1) Build sequences
    _, constructive, destructive = build_signals(args.N, args.mmax)

    # 2) FFTs
    freq_c, amp_c = rfft_spectrum(constructive)
    freq_d, amp_d = rfft_spectrum(destructive)

    # 3) Pick peaks
    fpk_c, apk_c = simple_peaks(freq_c, amp_c, k_top=args.peaks, fmin=1.0/args.N)
    fpk_d, apk_d = simple_peaks(freq_d, amp_d, k_top=args.peaks, fmin=1.0/args.N)

    # 4) Reference grid and matching
    refs_dict = generate_reference_dict(args.mmax, nyq=0.5)
    ref_freqs = list(refs_dict.keys())

    rows_c = match_peaks_to_refs(fpk_c, ref_freqs, args.N, tol_bins=3)
    rows_d = match_peaks_to_refs(fpk_d, ref_freqs, args.N, tol_bins=3)
    enrich_with_km(rows_c, refs_dict)
    enrich_with_km(rows_d, refs_dict)

    # 5) Save CSVs
    save_peak_map_csv("fft_peak_map_constructive.csv", rows_c)
    save_peak_map_csv("fft_peak_map_destructive.csv", rows_d)

    # 6) Print a quick summary
    print("\nTop mapped peaks — Constructive (freq ≈ k/m):")
    for r in rows_c[:20]:
        if r.get("match_m"):
            print(f"  {r['peak_freq']:.6f}  ≈  {r['match_k']}/{r['match_m']:>2} "
                  f"(err {r['bins_error']:.2f} bins, {r['ppm_error']:.1f} ppm)")
        else:
            print(f"  {r['peak_freq']:.6f}  (no match)")

    print("\nTop mapped peaks — Destructive (freq ≈ k/m):")
    for r in rows_d[:20]:
        if r.get("match_m"):
            print(f"  {r['peak_freq']:.6f}  ≈  {r['match_k']}/{r['match_m']:>2} "
                  f"(err {r['bins_error']:.2f} bins, {r['ppm_error']:.1f} ppm)")
        else:
            print(f"  {r['peak_freq']:.6f}  (no match)")

    print("\n✅ Done. Wrote: fft_peak_map_constructive.csv, fft_peak_map_destructive.csv")

if __name__ == "__main__":
    main()
