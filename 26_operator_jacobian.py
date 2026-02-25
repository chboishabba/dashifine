#!/usr/bin/env python3
# 27_operator_jacobian.py

import numpy as np
import argparse, os, glob, json
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import lstsq, eigvals, svd

try:
    import pandas as pd
except Exception:
    pd = None

def load_beta_trajectories(root):
    traj = []
    files = glob.glob(os.path.join(root, "**/*.npz"), recursive=True)
    files += glob.glob(os.path.join(root, "**/*.csv"), recursive=True)

    def add_traj(B):
        if B.ndim == 2 and B.shape[0] >= 3:
            traj.append(B.astype(np.float64))

    def load_csv(path):
        if pd is None:
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        cols = list(df.columns)
        candidates = []
        for prefix in [
            "beta_", "beta", "BETA_", "BETA",
            "coeff_", "coeff", "COEFF_", "COEFF",
            "b_", "b", "B_", "B",
        ]:
            colset = []
            for i in range(0, 64):
                c = f"{prefix}{i}"
                if c in cols:
                    colset.append(c)
                else:
                    break
            if len(colset) >= 2:
                candidates.append(colset)
        if not candidates:
            return
        colset = max(candidates, key=len)
        B = df[colset].to_numpy(dtype=np.float64)
        add_traj(B)

    for p in files:
        try:
            if p.endswith(".npz"):
                z = np.load(p)
                for k in z.files:
                    if "beta" in k.lower() or "coeff" in k.lower():
                        B = z[k]
                        add_traj(B)
            elif p.endswith(".csv"):
                load_csv(p)
        except:
            continue
    if not traj:
        raise RuntimeError("No contraction trajectories found.")
    return np.vstack(traj)

def estimate_jacobian(points, k=20):
    n, d = points.shape
    nbrs = NearestNeighbors(n_neighbors=min(k, n-1)).fit(points)
    J_list = []

    for i in range(n):
        x = points[i]
        dists, idx = nbrs.kneighbors([x])
        idx = idx[0][1:]
        X = points[idx] - x
        Y = np.roll(points, -1, axis=0)[idx] - x  # next-step approx

        # linear least squares
        try:
            J, _, _, _ = lstsq(X, Y, rcond=None)
            J_list.append(J.T)
        except:
            continue

    return np.array(J_list)

def contraction_metrics(Js):
    eigs = []
    svals = []
    for J in Js:
        eigs.append(eigvals(J))
        svals.append(svd(J, compute_uv=False))
    eigs = np.array(eigs)
    svals = np.array(svals)
    return eigs, svals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    P = load_beta_trajectories(args.beta_root)
    Js = estimate_jacobian(P)

    eigs, svals = contraction_metrics(Js)

    report = {
        "n_jacobians": int(len(Js)),
        "mean_spectral_radius": float(np.mean(np.max(np.abs(eigs), axis=1))),
        "mean_operator_norm": float(np.mean(np.max(svals, axis=1))),
        "contraction_fraction": float(np.mean(np.max(svals, axis=1) < 1.0))
    }

    with open(os.path.join(args.out, "jacobian_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    np.save(os.path.join(args.out, "jacobians.npy"), Js)

    print("Jacobian analysis complete.")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
