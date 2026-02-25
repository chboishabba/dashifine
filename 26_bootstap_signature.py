# bootstrap_signature.py

import numpy as np
from numpy.linalg import eigvals
import argparse, json, os

def classify(G):
    vals = np.real(eigvals(G))
    pos = int(np.sum(vals > 1e-6))
    neg = int(np.sum(vals < -1e-6))
    zero = int(len(vals) - pos - neg)
    return pos, neg, zero

def bootstrap(Js, n=100):
    sigs = []
    for _ in range(n):
        idx = np.random.choice(len(Js), len(Js), replace=True)
        Jb = Js[idx]
        G = np.eye(Jb.shape[1])
        for _ in range(50):
            G_new = np.zeros_like(G)
            for J in Jb:
                G_new += J.T @ G @ J
            G_new = G_new / len(Jb)
            tr = np.trace(G_new)
            if np.isfinite(tr) and abs(tr) > 1e-12:
                G_new = G_new / tr
            G = G_new
        sigs.append(classify(G))
    return sigs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jacobians", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Js = np.load(args.jacobians)
    sigs = bootstrap(Js)

    with open(os.path.join(args.out, "bootstrap_signatures.json"), "w") as f:
        json.dump(sigs, f)

    print("Bootstrap complete.")
