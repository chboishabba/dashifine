#!/usr/bin/env python3
# 26_dashi_proof_runner_v2.py
import argparse, os, json, glob, math, random
import numpy as np

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from ripser import ripser
except Exception as e:
    raise SystemExit("ripser is required: pip install ripser") from e

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def robust_load_lens(root, prefer_dim=None, verbose=False):
    """
    Returns:
      X: (N,D) float
      obs_list: list[str]
      per_obs: dict[obs] -> (n_obs, D)
    """
    hits = []
    # gather files
    files = []
    files += glob.glob(os.path.join(root, "**/*.npz"), recursive=True)
    files += glob.glob(os.path.join(root, "**/*.npy"), recursive=True)
    files += glob.glob(os.path.join(root, "**/*.csv"), recursive=True)

    def add_obs(obsname, A):
        A = np.asarray(A)
        if A.ndim == 1:
            A = A[None, :]
        if A.ndim != 2 or A.shape[0] < 2:
            return
        if not np.isfinite(A).all():
            return
        hits.append((obsname, A.astype(np.float64)))

    # helper: infer obs name from path (dir under root)
    def infer_obs(path):
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        # heuristic: first directory is observable
        return parts[0] if len(parts) > 1 else os.path.splitext(parts[0])[0]

    # NPZ
    for p in files:
        obs = infer_obs(p)
        if p.endswith(".npz"):
            try:
                z = np.load(p, allow_pickle=True)
                for k in z.files:
                    A = z[k]
                    if isinstance(A, np.ndarray) and A.ndim in (1,2):
                        if A.ndim == 1 and A.shape[0] >= 2:
                            add_obs(obs, A)
                        if A.ndim == 2 and A.shape[0] >= 2 and A.shape[1] >= 2:
                            add_obs(obs, A)
            except Exception:
                continue
        elif p.endswith(".npy"):
            try:
                A = np.load(p, allow_pickle=True)
                if isinstance(A, np.ndarray) and A.ndim in (1,2):
                    add_obs(obs, A)
            except Exception:
                continue
        elif p.endswith(".csv") and pd is not None:
            try:
                df = pd.read_csv(p)
                cols = list(df.columns)
                # choose best lens-like column set
                candidates = []
                for prefix in ["lens_", "lens", "LENS_", "LENS",
               "L_", "L",
               "beta_", "beta",
               "coeff_", "coeff",
               "c_", "c",
               "b_", "b"]:
                    colset = []
                    for i in range(0, 64):
                        c = f"{prefix}{i}"
                        if c in cols:
                            colset.append(c)
                        else:
                            break
                    if len(colset) >= 2:
                        candidates.append(colset)
                if candidates:
                    colset = max(candidates, key=len)
                    A = df[colset].to_numpy(dtype=np.float64)
                    add_obs(obs, A)
            except Exception:
                continue

    if not hits:
        raise RuntimeError(
            "No usable lens arrays found. Run 26_lens_inspect.py and check what files/keys exist."
        )

    # choose a consistent D by majority vote (and prefer_dim if given)
    dims = [A.shape[1] for _, A in hits]
    unique, counts = np.unique(dims, return_counts=True)
    dim_mode = int(unique[np.argmax(counts)])
    if prefer_dim is not None:
        # if prefer_dim exists among hits, use it
        if prefer_dim in unique:
            dim_mode = int(prefer_dim)

    filtered = [(obs, A) for obs, A in hits if A.shape[1] == dim_mode]
    if verbose:
        print(f"Found {len(hits)} candidates; using D={dim_mode} with {len(filtered)} arrays.")

    all_rows = []
    per_obs = {}
    obs_list = []
    for obs, A in filtered:
        all_rows.append(A)
        if obs not in per_obs:
            per_obs[obs] = A.shape
            obs_list.append(obs)
    X = np.vstack(all_rows)
    # de-dup exact rows (optional; helps PH stability)
    X = np.unique(X, axis=0)
    return X, obs_list, per_obs

def whiten(X, eps=1e-6):
    mu = X.mean(axis=0, keepdims=True)
    C = np.cov(X.T)
    C = C + eps*np.eye(C.shape[0])
    L = np.linalg.cholesky(C)
    Xw = np.linalg.solve(L, (X - mu).T).T
    return Xw, mu.squeeze(), C

def twoNN_dim(X, k=2):
    # TwoNN estimator (Facco et al): ratio r2/r1
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=max(3, k+1)).fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1 = dists[:,1]
    r2 = dists[:,2]
    # avoid zeros
    mask = (r1 > 1e-12) & (r2 > 1e-12)
    mu = (r2[mask] / r1[mask])
    mu = np.sort(mu)
    n = len(mu)
    if n < 10:
        return float("nan")
    y = -np.log(1.0 - (np.arange(1, n+1) - 0.5)/n)
    x = np.log(mu)
    # slope through origin gives dim
    dim = float((x @ y) / (x @ x))
    return dim

def mle_idim(X, k=10):
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    k = min(k, n-1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    d = dists[:,1:]  # exclude self
    rk = d[:, -1]
    # avoid zero
    eps = 1e-12
    ratios = np.log((rk[:,None] + eps) / (d + eps))
    idim = (k-1) / np.sum(ratios[:, :-1], axis=1)
    # robust average
    return float(np.median(idim[np.isfinite(idim)]))

def pca_evr(X):
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(X.shape[1], X.shape[0]-1)).fit(X)
    evr = p.explained_variance_ratio_
    return evr

def diffusion_eigs(X, n_eigs=10, k=10, alpha=1.0):
    # simple diffusion map: kernel on kNN distances (Gaussian)
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    k = min(k, n-1)
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, idx = nn.kneighbors(X)
    sig = np.median(dists[:,1:]) + 1e-12
    W = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for jpos in range(1, k+1):
            j = idx[i, jpos]
            dij = dists[i, jpos]
            w = math.exp(-(dij*dij)/(2*sig*sig))
            W[i,j] = max(W[i,j], w)
            W[j,i] = max(W[j,i], w)
    # normalize
    d = W.sum(axis=1)
    d[d==0] = 1.0
    Dinv = np.diag(1.0 / (d**alpha))
    K = Dinv @ W @ Dinv
    d2 = K.sum(axis=1)
    d2[d2==0] = 1.0
    P = (K.T / d2).T  # row-stochastic
    # eigs
    vals = np.linalg.eigvals(P)
    vals = np.real(vals)
    vals = np.sort(vals)[::-1]
    return vals[:min(n_eigs, len(vals))]

def isomap_resid(X, dims=6, k=10):
    from sklearn.manifold import Isomap
    from sklearn.metrics import pairwise_distances
    n = X.shape[0]
    k = min(k, n-1)
    # residual variance = 1 - R^2 between geodesic distances and embedded distances
    iso = Isomap(n_neighbors=k, n_components=min(dims, X.shape[1], n-1))
    Y = iso.fit_transform(X)
    G = iso.dist_matrix_
    G = (G + G.T)/2
    res = []
    for d in range(1, Y.shape[1]+1):
        Yd = Y[:, :d]
        Ed = pairwise_distances(Yd)
        # correlate flattened upper triangle
        iu = np.triu_indices_from(G, k=1)
        g = G[iu]
        e = Ed[iu]
        if np.std(g) < 1e-12 or np.std(e) < 1e-12:
            res.append(float("nan"))
            continue
        R = np.corrcoef(g, e)[0,1]
        res.append(1.0 - R*R)
    return res

def ph_max_lifetime(X, maxdim=2, thresh=None, nperm=None, distance_matrix=False):
    # ripser; return max lifetime in H1, H2
    kw = {"maxdim": maxdim}
    if thresh is not None:
        kw["thresh"] = float(thresh)
    if nperm is not None:
        n = int(X.shape[0])
        nperm = min(int(nperm), n)
        if nperm >= 2:
            kw["n_perm"] = nperm
    if distance_matrix:
        kw["distance_matrix"] = True
    dgms = ripser(X, **kw)["dgms"]
    def maxlife(dgm):
        if dgm is None or len(dgm) == 0:
            return 0.0
        b = dgm[:,0]
        d = dgm[:,1]
        # inf deaths -> ignore or cap?
        mask = np.isfinite(d)
        if not np.any(mask):
            return 0.0
        return float(np.max(d[mask] - b[mask]))
    H1 = maxlife(dgms[1]) if len(dgms) > 1 else 0.0
    H2 = maxlife(dgms[2]) if len(dgms) > 2 else 0.0
    return H1, H2, dgms

def gaussian_null(X, reps=50, seed=0):
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0)
    C = np.cov(X.T)
    C = (C + C.T)/2
    out = []
    for r in range(reps):
        G = rng.multivariate_normal(mu, C, size=X.shape[0])
        out.append(G)
    return out

def colshuffle_null(X, reps=50, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for r in range(reps):
        Y = X.copy()
        for j in range(Y.shape[1]):
            rng.shuffle(Y[:,j])
        out.append(Y)
    return out

def safe_traj_ph(traj_points, max_points=2000, chunk=500, seed=0):
    """
    Prevent SIGKILL by downsampling and chunking.
    traj_points: (T*N, D) possibly huge
    """
    rng = np.random.default_rng(seed)
    P = traj_points
    if P.shape[0] > max_points:
        idx = rng.choice(P.shape[0], size=max_points, replace=False)
        P = P[idx]
    # ripser can still be heavy; use n_perm for approx if needed
    H1, H2, _ = ph_max_lifetime(P, maxdim=2, thresh=None, nperm=200)
    return H1, H2, P.shape[0]

def load_beta_flows(beta_root, want_obs=None):
    """
    Tries to load per-observable contraction trajectory point clouds from hepdata_dashi_native output.
    Accepts:
      - npz with arrays named like beta / betas / coeff / coeffs
      - csv with columns beta_0..beta_k
    Returns dict[obs] -> (T, d) beta trajectory
    """
    out = {}
    if beta_root is None or not os.path.isdir(beta_root):
        return out
    files = []
    files += glob.glob(os.path.join(beta_root, "**/*.npz"), recursive=True)
    files += glob.glob(os.path.join(beta_root, "**/*.csv"), recursive=True)

    def infer_obs(path):
        rel = os.path.relpath(path, beta_root)
        parts = rel.split(os.sep)
        return parts[0] if len(parts) > 1 else os.path.splitext(parts[0])[0]

    for p in sorted(files):
        obs = infer_obs(p)
        if want_obs and obs not in want_obs:
            continue
        if p.endswith(".npz"):
            try:
                z = np.load(p, allow_pickle=True)
                for k in z.files:
                    A = z[k]
                    if isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[0] >= 3 and A.shape[1] >= 2:
                        lk = k.lower()
                        if any(s in lk for s in ["beta", "coeff", "cvec", "params"]):
                            out[obs] = A.astype(np.float64)
                            break
            except Exception:
                continue
        elif p.endswith(".csv") and pd is not None:
            try:
                df = pd.read_csv(p)
                cols = list(df.columns)
                cand = []
                for prefix in ["beta_", "beta", "coeff_", "coeff"]:
                    colset = []
                    for i in range(0, 128):
                        c = f"{prefix}{i}"
                        if c in cols:
                            colset.append(c)
                        else:
                            break
                    if len(colset) >= 2:
                        cand.append(colset)
                if cand:
                    colset = max(cand, key=len)
                    A = df[colset].to_numpy(dtype=np.float64)
                    if A.shape[0] >= 3:
                        out[obs] = A
            except Exception:
                continue
    return out

def zscore(x, mu, sd):
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float((x - mu)/sd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-root", required=True)
    ap.add_argument("--beta-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--null-reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefer-dim", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    X, obs_list, per_obs = robust_load_lens(args.lens_root, prefer_dim=args.prefer_dim, verbose=args.verbose)
    n, d = X.shape
    print(f"Loaded lens space: n={n} d={d}")
    print(f"Observables used: {sorted(set(obs_list))}")

    # RAW metrics
    raw = {}
    raw["TwoNN"] = twoNN_dim(X)
    raw["MLE"] = mle_idim(X)
    evr = pca_evr(X)
    raw["PCA_evr"] = evr.tolist()
    raw["PCA_cum3"] = float(np.sum(evr[:3])) if len(evr) >= 3 else float(np.sum(evr))
    raw["diffusion_eigs"] = diffusion_eigs(X, n_eigs=min(10,d), k=min(10, n-1)).tolist()
    raw["isomap_resid"] = isomap_resid(X, dims=min(6, d), k=min(10, n-1))
    H1, H2, dgms = ph_max_lifetime(X, maxdim=2, thresh=None, nperm=None)
    raw["H1"] = H1
    raw["H2"] = H2

    # WHITEN
    Xw, mu, C = whiten(X)
    white = {}
    white["TwoNN"] = twoNN_dim(Xw)
    white["MLE"] = mle_idim(Xw)
    evr_w = pca_evr(Xw)
    white["PCA_evr"] = evr_w.tolist()
    white["PCA_cum3"] = float(np.sum(evr_w[:3])) if len(evr_w) >= 3 else float(np.sum(evr_w))
    white["diffusion_eigs"] = diffusion_eigs(Xw, n_eigs=min(10,d), k=min(10, n-1)).tolist()
    white["isomap_resid"] = isomap_resid(Xw, dims=min(6, d), k=min(10, n-1))
    H1w, H2w, dgms_w = ph_max_lifetime(Xw, maxdim=2, thresh=None, nperm=None)
    white["H1"] = H1w
    white["H2"] = H2w

    # Nulls
    reps = int(args.null_reps)
    gauss_raw = []
    gauss_dim = []
    gauss = gaussian_null(X, reps=reps, seed=args.seed+1)
    for G in gauss:
        h1, h2, _ = ph_max_lifetime(G, maxdim=2, nperm=200)
        gauss_raw.append(h1)
        gauss_dim.append(twoNN_dim(G))
    gauss_raw = np.array(gauss_raw, dtype=np.float64)
    gauss_dim = np.array(gauss_dim, dtype=np.float64)

    # Z vs Gaussian (RAW)
    z_h1 = zscore(raw["H1"], float(np.mean(gauss_raw)), float(np.std(gauss_raw)))
    z_dim = zscore(raw["TwoNN"], float(np.mean(gauss_dim)), float(np.std(gauss_dim)))

    # Trajectory PH (safe)
    beta = load_beta_flows(args.beta_root, want_obs=None)
    traj = {}
    if beta:
        # stack all trajectories
        P = []
        for obs, B in beta.items():
            if B.ndim == 2 and B.shape[0] >= 3 and np.isfinite(B).all():
                P.append(B)
        if P:
            P = np.vstack(P)
            th1, th2, used = safe_traj_ph(P, max_points=2000, seed=args.seed+7)
            traj = {"H1": th1, "H2": th2, "n_used": used}
    else:
        traj = {"H1": None, "H2": None, "n_used": 0}

    report = {
        "inputs": {
            "lens_root": os.path.abspath(args.lens_root),
            "beta_root": os.path.abspath(args.beta_root) if args.beta_root else None,
            "n": int(n), "d": int(d),
            "observables": sorted(set(obs_list)),
            "prefer_dim": args.prefer_dim,
            "null_reps": reps,
            "seed": args.seed,
        },
        "raw": raw,
        "whitened": white,
        "null_gaussian_raw": {
            "H1_mean": float(np.mean(gauss_raw)),
            "H1_std": float(np.std(gauss_raw)),
            "TwoNN_mean": float(np.mean(gauss_dim)),
            "TwoNN_std": float(np.std(gauss_dim)),
        },
        "z_scores_vs_gaussian_raw": {
            "H1": z_h1,
            "TwoNN": z_dim,
        },
        "trajectory_ph_beta_space": traj,
        "notes": [
            "Lens vectors auto-detected from npz/npy/csv; if you want strict key selection, use 26_lens_inspect.py to locate the exact key/file.",
            "Trajectory PH uses safe downsampling + n_perm approximation to avoid SIGKILL.",
        ],
    }

    out_json = os.path.join(args.out, "proof_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== HEADLINE ===")
    print(f"RAW:   TwoNN={raw['TwoNN']:.3f}  MLE={raw['MLE']:.3f}  PCA_cum3={raw['PCA_cum3']:.3f}  H1={raw['H1']:.4f}  H2={raw['H2']:.4f}")
    print(f"WHITE: TwoNN={white['TwoNN']:.3f} MLE={white['MLE']:.3f} PCA_cum3={white['PCA_cum3']:.3f} H1={white['H1']:.4f} H2={white['H2']:.4f}")
    print(f"Null(Gauss RAW): TwoNN={report['null_gaussian_raw']['TwoNN_mean']:.3f}±{report['null_gaussian_raw']['TwoNN_std']:.3f}  H1={report['null_gaussian_raw']['H1_mean']:.4f}±{report['null_gaussian_raw']['H1_std']:.4f}")
    print(f"Z: TwoNN vs Gauss = {z_dim:.2f}σ ; H1 vs Gauss = {z_h1:.2f}σ")
    if traj["H1"] is not None:
        print(f"Trajectory(beta) PH: H1={traj['H1']:.4f} H2={traj['H2']:.4f} (n_used={traj['n_used']})")
    print(f"\nWrote: {out_json}")

if __name__ == "__main__":
    main()
