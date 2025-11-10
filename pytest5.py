#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def alignment_strength(n: int, mmax: int):
    x = np.arange(0, n + 1)
    A = np.zeros_like(x, dtype=np.int32)
    for m in range(1, mmax + 1):
        A += (x % m == 0)
    return x, A

def moving_avg(y, win):
    if win <= 1:
        return y
    w = np.ones(win) / win
    return np.convolve(y, w, mode="same")

def main():
    ap = argparse.ArgumentParser(description="Compute alignment strength (# of moduli dividing x)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--mmax", type=int, default=11)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--csv", default="alignment_strength.csv")
    ap.add_argument("--decimate", type=int, default=1)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    x, A = alignment_strength(args.n, args.mmax)
    df = pd.DataFrame({"x": x, "alignment_strength": A})
    df.to_csv(args.csv, index=False)
    print(f"✅ computed 0..{args.n} (mod 1..{args.mmax}); wrote {args.csv}")

    # --- print top bins (console table) ---
    top = df.sort_values("alignment_strength", ascending=False).head(args.top)
    print(f"\nTop {args.top} alignment points:")
    print(top.to_string(index=False))

    # --- optional plotting (line only) ---
    if args.plot:
        step = max(1, args.decimate)
        xs = x[::step]
        As = A[::step]
        As_smooth = moving_avg(As, args.smooth)

        plt.figure(figsize=(12, 5))
        plt.plot(xs, As_smooth, lw=0.9)
        plt.title(f"A(x): # of moduli (1..{args.mmax}) dividing x  [decimate={step}, smooth={args.smooth}]")
        plt.xlabel("x")
        plt.ylabel("A(x)")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
