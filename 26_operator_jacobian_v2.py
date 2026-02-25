#!/usr/bin/env python3
# 26_operator_jacobian_v2.py
import os, glob, json, argparse
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

def _load_beta_from_npz(path: str):
    z = np.load(path)
    keys = list(z.files)
    # try common keys first
    pref = [k for k in keys if ("beta" in k.lower() or "coeff" in k.lower())]
    for k in pref + keys:
        B = z[k]
        if isinstance(B, np.ndarray) and B.ndim == 2 and B.shape[0] >= 3:
            return B.astype(np.float64)
    return None

def _load_beta_from_csv(path: str):
    if pd is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = list(df.columns)

    candidates = []
    for prefix in ["beta_", "beta", "BETA_", "BETA", "coeff_", "coeff", "COEFF_","COEFF","b_","b","B_","B"]:
        colset = []
        for i in range(0, 256):
            c = f"{prefix}{i}"
            if c in cols:
                colset.append(c)
            else:
                break
        if len(colset) >= 2:
            candidates.append(colset)

    # also allow unnamed numeric columns if present
    if not candidates:
        numeric_cols = [c for c in cols if c.strip().isdigit()]
        if len(numeric_cols) >= 2:
            numeric_cols = sorted(numeric_cols, key=lambda s: int(s))
            candidates.append(numeric_cols)

    if not candidates:
        return None

    colset = max(candidates, key=len)
    B = df[colset].to_numpy(dtype=np.float64)
    if B.ndim == 2 and B.shape[0] >= 3:
        return B
    return None

def load_trajectories(beta_root: str):
    files = []
    files += glob.glob(os.path.join(beta_root, "**/*.npz"), recursive=True)
    files += glob.glob(os.path.join(beta_root, "**/*.csv"), recursive=True)
    files = sorted(set(files))

    trajs = []
    for p in files:
        try:
            if p.endswith(".npz"):
                B = _load_beta_from_npz(p)
            else:
                B = _load_beta_from_csv(p)
            if B is None:
                continue
            # require time × dim
            if B.ndim == 2 and B.shape[0] >= 3 and B.shape[1] >= 1:
                trajs.append(B)
        except Exception:
            continue

    if not trajs:
        raise RuntimeError(f"No trajectories found under: {beta_root}")
    # ensure common dimension by truncation to min-d
    dmin = min(t.shape[1] for t in trajs)
    trajs = [t[:, :dmin].astype(np.float64) for t in trajs]
    return trajs

def estimate_fixed_point(trajs, tail: int):
    # per-trajectory tail mean, then global mean
    tails = []
    for T in trajs:
        k = min(tail, T.shape[0])
        tails.append(np.mean(T[-k:, :], axis=0))
    xstar = np.mean(np.stack(tails, axis=0), axis=0)
    return xstar

def fit_global_J(trajs, xstar, max_pairs=None):
    Xs, Ys = [], []
    for T in trajs:
        # build consecutive pairs
        D0 = T[:-1, :] - xstar
        D1 = T[1:,  :] - xstar
        Xs.append(D0)
        Ys.append(D1)
    X = np.vstack(Xs)
    Y = np.vstack(Ys)

    # optional subsample (for huge runs)
    if max_pairs is not None and X.shape[0] > max_pairs:
        idx = np.random.default_rng(0).choice(X.shape[0], size=max_pairs, replace=False)
        X = X[idx]
        Y = Y[idx]

    # solve Y ≈ X @ J^T  => J^T = lstsq(X, Y)
    JT, *_ = np.linalg.lstsq(X, Y, rcond=None)
    J = JT.T
    return J

def per_traj_J(trajs, xstar):
    Js = []
    for T in trajs:
        X = T[:-1, :] - xstar
        Y = T[1:,  :] - xstar
        if X.shape[0] < X.shape[1] + 1:
            continue
        JT, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Js.append(JT.T)
    if not Js:
        return None
    return np.stack(Js, axis=0)

def op_metrics(J):
    # spectral radius and operator 2-norm (largest singular value)
    svals = np.linalg.svd(J, compute_uv=False)
    opnorm = float(np.max(svals))
    eig = np.linalg.eigvals(J)
    spr = float(np.max(np.abs(eig)))
    return spr, opnorm, eig, svals

def metric_op_norm(J, G, eps=1e-12):
    # sqrt( lambda_max( G^{-1/2} J^T G J G^{-1/2} ) )
    G = 0.5 * (G + G.T)
    G = G + eps * np.eye(G.shape[0])
    L = np.linalg.cholesky(G)
    JT_G_J = J.T @ G @ J
    # A = L^{-1} (JT_G_J) L^{-T}
    X = np.linalg.solve(L, JT_G_J)
    A = np.linalg.solve(L, X.T).T
    A = 0.5 * (A + A.T)
    lam_max = float(np.max(np.linalg.eigvalsh(A)))
    return float(np.sqrt(max(lam_max, 0.0)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tail", type=int, default=5, help="tail steps used to estimate x*")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    trajs = load_trajectories(args.beta_root)
    xstar = estimate_fixed_point(trajs, tail=args.tail)
    J = fit_global_J(trajs, xstar, max_pairs=args.max_pairs)

    spr, opnorm, eig, svals = op_metrics(J)
    contraction = (opnorm < 1.0)

    report = {
        "n_traj": int(len(trajs)),
        "dim": int(xstar.shape[0]),
        "tail": int(args.tail),
        "global_spectral_radius": spr,
        "global_operator_norm": opnorm,
        "global_contractive": bool(contraction),
    }

    np.save(os.path.join(args.out, "xstar.npy"), xstar)
    np.save(os.path.join(args.out, "J_global.npy"), J)
    np.save(os.path.join(args.out, "eig_global.npy"), eig)
    np.save(os.path.join(args.out, "svals_global.npy"), svals)

    # also per-trajectory Jacobians (optional but very useful)
    Js = per_traj_J(trajs, xstar)
    if Js is not None:
        # per-traj norms
        smax = np.max(np.linalg.svd(Js, compute_uv=False), axis=1)
        report["per_traj_operator_norm_mean"] = float(np.mean(smax))
        report["per_traj_contraction_fraction"] = float(np.mean(smax < 1.0))
        np.save(os.path.join(args.out, "J_per_traj.npy"), Js)

    # metric-based contraction (if G available)
    G_path = os.path.join(args.out, "G_lyapunov.npy")
    if os.path.exists(G_path):
        G = np.load(G_path)
        report["global_metric_op_norm_G"] = metric_op_norm(J, G)
        report["global_contractive_in_G"] = bool(report["global_metric_op_norm_G"] < 1.0)
        if Js is not None:
            normsG = [metric_op_norm(Jt, G) for Jt in Js]
            report["per_traj_metric_op_norm_G_mean"] = float(np.mean(normsG))
            report["per_traj_contraction_fraction_G"] = float(np.mean(np.array(normsG) < 1.0))

    with open(os.path.join(args.out, "jacobian_report_v2.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
