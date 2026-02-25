#!/usr/bin/env python3
# 26_quadratic_fit_v2.py
import os, json, argparse
import numpy as np

def solve_discrete_lyapunov(J, iters=2000, tol=1e-10):
    d = J.shape[0]
    G = np.eye(d, dtype=np.float64)
    I = np.eye(d, dtype=np.float64)

    # fixed point iteration: G_{k+1} = I + J^T G_k J
    for _ in range(iters):
        G_new = I + J.T @ G @ J
        G_new = 0.5 * (G_new + G_new.T)
        if np.linalg.norm(G_new - G, ord="fro") < tol:
            G = G_new
            break
        G = G_new
    return 0.5 * (G + G.T)

def classify_signature(G, eps=1e-8):
    vals = np.linalg.eigvalsh(G)
    pos = int(np.sum(vals > eps))
    neg = int(np.sum(vals < -eps))
    zer = int(len(vals) - pos - neg)
    return {"positive": pos, "negative": neg, "zero": zer}, vals.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", required=True, help="path to J_global.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    J = np.load(args.J)
    G = solve_discrete_lyapunov(J, iters=args.iters, tol=args.tol)
    sig, eigs = classify_signature(G)

    np.save(os.path.join(args.out, "G_lyapunov.npy"), G)
    rep = {"signature": sig, "eigenvalues": eigs}
    with open(os.path.join(args.out, "signature_report_v2.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("Signature:", sig)

if __name__ == "__main__":
    main()
