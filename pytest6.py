#!/usr/bin/env python3
import argparse, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def parse_series(s: str):
    """
    Parse mmax series like: '2,3,4,5' or '2..11' or mixed '2,3,5..9,11'
    """
    out = []
    for part in s.split(","):
        part = part.strip()
        if ".." in part:
            a, b = part.split("..")
            a, b = int(a), int(b)
            out.extend(range(min(a,b), max(a,b)+1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    return out

def alignment_strength(n: int, mmax: int):
    x = np.arange(0, n + 1)
    A = np.zeros_like(x, dtype=np.int32)
    for m in range(1, mmax + 1):
        A += (x % m == 0)
    return x, A

def moving_avg(y: np.ndarray, win: int):
    if win <= 1: return y
    w = np.ones(win, dtype=float) / win
    return np.convolve(y, w, mode="same")

def multiples_up_to(val: int, n: int):
    if val <= 0: return []
    return list(range(0, n + 1, val))

def main():
    ap = argparse.ArgumentParser(description="Overlay alignment-strength curves for multiple mmax values")
    ap.add_argument("--n", type=int, default=1000, help="range 0..n")
    ap.add_argument("--series", default="2..11",
                    help="mmax values to overlay, e.g. '2..11' or '2,3,5..9,11'")
    ap.add_argument("--normalize", action="store_true",
                    help="plot A(x)/mmax instead of raw A(x)")
    ap.add_argument("--smooth", type=int, default=1, help="moving average window (samples)")
    ap.add_argument("--decimate", type=int, default=1, help="plot every k-th x for speed")
    ap.add_argument("--mark_lcms", nargs="*", type=int,
                    help="LCM values to draw vertical lines at (e.g., 30 60 84 90)")
    ap.add_argument("--save", default="", help="path to save image (e.g., 'overlay.png'); no window shown if set")
    args = ap.parse_args()

    # If saving to file, avoid interactive backend
    if args.save:
        matplotlib.use("Agg")

    mmax_list = parse_series(args.series)
    x_full = np.arange(0, args.n + 1, dtype=int)

    plt.figure(figsize=(12, 5))
    for mmax in mmax_list:
        x, A = alignment_strength(args.n, mmax)
        step = max(1, args.decimate)
        xs = x[::step]
        ys = A[::step].astype(float)
        if args.normalize:
            ys /= float(mmax)
        ys = moving_avg(ys, args.smooth)
        plt.plot(xs, ys, linewidth=1.0, label=f"mmax={mmax}")

    plt.title(f"Alignment strength over 0..{args.n}"
              + (" (normalized)" if args.normalize else "")
              + (f"  [decimate={args.decimate}, smooth={args.smooth}]" if (args.decimate>1 or args.smooth>1) else ""))
    plt.xlabel("x")
    plt.ylabel("A(x)" + ("/mmax" if args.normalize else ""))

    # Optional LCM markers
    if args.mark_lcms:
        for v in args.mark_lcms:
            for xi in multiples_up_to(v, args.n):
                plt.axvline(xi, linestyle="--", linewidth=0.6, alpha=0.25)

    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper right", ncols=min(len(mmax_list), 4), fontsize=9)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=180)
    else:
        plt.show()

if __name__ == "__main__":
    main()
