#!/usr/bin/env python3
# 28_quadratic_fit.py

import numpy as np
import argparse, os, json
from numpy.linalg import eigvals

def solve_invariant_quadratic(Js, iters=100, damp=0.2):
    d = Js.shape[1]
    G = np.eye(d, dtype=np.float64)

    # normalize operators to avoid blow-up when Js are expansive
    Js_norm = []
    for J in Js:
        try:
            smax = np.linalg.svd(J, compute_uv=False)[0]
        except Exception:
            smax = np.linalg.norm(J, ord=2)
        scale = max(1.0, float(smax))
        Js_norm.append(J / scale)

    # iterative average projection with damping + rescale
    for _ in range(iters):
        G_new = np.zeros_like(G)
        for J in Js_norm:
            G_new += J.T @ G @ J
        G_new /= max(1, len(Js_norm))
        G = (1.0 - damp) * G + damp * G_new
        G = (G + G.T) / 2
        # rescale to keep magnitudes bounded
        tr = np.trace(G)
        if np.isfinite(tr) and abs(tr) > 1e-12:
            G = G / tr
        else:
            fn = np.linalg.norm(G)
            if np.isfinite(fn) and fn > 1e-12:
                G = G / fn

    return (G + G.T) / 2

def classify_signature(G):
    vals = np.real(eigvals(G))
    pos = np.sum(vals > 1e-6)
    neg = np.sum(vals < -1e-6)
    zero = len(vals) - pos - neg
    return pos, neg, zero, vals.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jacobians", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    Js = np.load(args.jacobians)
    G = solve_invariant_quadratic(Js)
    pos, neg, zero, vals = classify_signature(G)

    report = {
        "signature": {
            "positive": int(pos),
            "negative": int(neg),
            "zero": int(zero)
        },
        "eigenvalues": vals
    }

    np.save(os.path.join(args.out, "quadratic_G.npy"), G)
    with open(os.path.join(args.out, "signature_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Signature:", report["signature"])

if __name__ == "__main__":
    main()
