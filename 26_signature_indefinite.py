#!/usr/bin/env python3
# 26_signature_indefinite.py

import numpy as np
import argparse, json, os

def solve_homogeneous(J):
    d = J.shape[0]
    # We solve (J^T ⊗ J^T - I) vec(G) = 0
    A = np.kron(J.T, J.T) - np.eye(d*d)
    # Nullspace via SVD
    U, S, Vt = np.linalg.svd(A)
    tol = 1e-8
    null_mask = (S < tol)
    if not np.any(null_mask):
        return None
    vecG = Vt.T[:, null_mask][:,0]
    G = vecG.reshape((d,d))
    G = 0.5*(G + G.T)
    return G

def classify(G):
    vals = np.linalg.eigvalsh(G)
    pos = int(np.sum(vals > 1e-8))
    neg = int(np.sum(vals < -1e-8))
    zero = int(len(vals) - pos - neg)
    return {"positive":pos,"negative":neg,"zero":zero}, vals.tolist()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    J = np.load(args.J)

    G = solve_homogeneous(J)
    if G is None:
        print("No invariant quadratic found.")
        exit()

    sig, eigs = classify(G)
    np.save(os.path.join(args.out,"G_indefinite.npy"),G)

    with open(os.path.join(args.out,"signature_indefinite.json"),"w") as f:
        json.dump({"signature":sig,"eigenvalues":eigs},f,indent=2)

    print("Signature:",sig)
