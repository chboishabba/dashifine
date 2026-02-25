#!/usr/bin/env python3
# 26_lens_inspect.py
import argparse, os, json, glob
import numpy as np

def _looks_like_lens_matrix(A, dmin=2, dmax=256):
    if not isinstance(A, np.ndarray): return False
    if A.ndim == 1:
        return dmin <= A.shape[0] <= dmax
    if A.ndim == 2:
        return dmin <= A.shape[1] <= dmax and A.shape[0] >= 2
    return False

def inspect_npz(path):
    out = []
    try:
        z = np.load(path, allow_pickle=True)
        for k in z.files:
            A = z[k]
            if _looks_like_lens_matrix(A):
                out.append((k, tuple(A.shape), str(A.dtype)))
    except Exception as e:
        out.append(("__ERROR__", (), str(e)))
    return out

def inspect_npy(path):
    out = []
    try:
        A = np.load(path, allow_pickle=True)
        if _looks_like_lens_matrix(A):
            out.append(("", tuple(A.shape), str(A.dtype)))
    except Exception as e:
        out.append(("__ERROR__", (), str(e)))
    return out

def inspect_csv(path):
    out = []
    try:
        import pandas as pd
        df = pd.read_csv(path)
        cols = list(df.columns)
        # candidate column sets
        cand_sets = []
        # lens_0..lens_9
        for prefix in ["lens_", "lens", "LENS_", "LENS",
               "L_", "L",
               "beta_", "beta",
               "coeff_", "coeff",
               "c_", "c",
               "b_", "b"]:
            s = [f"{prefix}{i}" for i in range(0, 64)]
            if all(c in cols for c in s[:10]):
                # take maximal contiguous
                k = 0
                while k < len(s) and s[k] in cols: k += 1
                cand_sets.append(s[:k])
        if cand_sets:
            for s in cand_sets:
                out.append(("cols:" + ",".join(s[:min(10,len(s))]) + ("..." if len(s)>10 else ""),
                            (len(df), len(s)),
                            "float?"))
    except Exception as e:
        out.append(("__ERROR__", (), str(e)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-root", required=True, help="e.g. hepdata_to_dashi")
    ap.add_argument("--out", default="lens_manifest.json")
    args = ap.parse_args()

    root = args.lens_root
    files = []
    files += glob.glob(os.path.join(root, "**/*.npz"), recursive=True)
    files += glob.glob(os.path.join(root, "**/*.npy"), recursive=True)
    files += glob.glob(os.path.join(root, "**/*.csv"), recursive=True)

    manifest = {"root": os.path.abspath(root), "hits": []}
    for p in sorted(files):
        rel = os.path.relpath(p, root)
        if p.endswith(".npz"):
            hits = inspect_npz(p)
            for k, shp, dt in hits:
                if k == "__ERROR__":
                    manifest["hits"].append({"file": rel, "kind": "npz", "key": k, "shape": shp, "dtype": dt})
                else:
                    manifest["hits"].append({"file": rel, "kind": "npz", "key": k, "shape": shp, "dtype": dt})
        elif p.endswith(".npy"):
            hits = inspect_npy(p)
            for k, shp, dt in hits:
                manifest["hits"].append({"file": rel, "kind": "npy", "key": "", "shape": shp, "dtype": dt})
        elif p.endswith(".csv"):
            hits = inspect_csv(p)
            for k, shp, dt in hits:
                manifest["hits"].append({"file": rel, "kind": "csv", "key": k, "shape": shp, "dtype": dt})

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.out} with {len(manifest['hits'])} candidate lens arrays.")
    # print top few
    for h in manifest["hits"][:30]:
        print(h)

if __name__ == "__main__":
    main()
