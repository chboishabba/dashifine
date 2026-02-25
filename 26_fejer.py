#!/usr/bin/env python3
import argparse, math
import numpy as np
import pandas as pd

def monotone_stats(vals, eps=1e-12):
    vals = np.asarray(vals, dtype=float)
    ok = (vals[1:] <= vals[:-1] + eps)
    frac = float(ok.mean()) if len(ok) else 1.0
    if len(ok):
        viol = vals[1:] - vals[:-1]
        max_viol = float(viol.max())
        arg = int(viol.argmax()) + 1
    else:
        max_viol, arg = 0.0, 0
    return frac, max_viol, arg

def parse_signature(sig_str, dim):
    """
    sig_str: e.g. "+++ -" or "3,1" or "+,+,+,-"
    Returns diagonal mask m_i in {+1,-1} length dim.
    """
    s = sig_str.strip()
    if "," in s and all(t.strip().isdigit() for t in s.split(",")):
        p, n = [int(t.strip()) for t in s.split(",")]
        m = np.array([+1]*p + [-1]*n, dtype=float)
        if len(m) != dim:
            raise ValueError(f"signature {p},{n} gives dim={len(m)} but need dim={dim}")
        return m
    # token form
    toks = [t for t in s.replace(" ", "").replace(";", ",").split(",") if t]
    if len(toks) == 1 and set(toks[0]).issubset({"+","-"}):
        toks = list(toks[0])
    if len(toks) != dim:
        raise ValueError(f"signature tokens {toks} gives dim={len(toks)} but need dim={dim}")
    m = np.array([+1 if t=="+" else -1 for t in toks], dtype=float)
    return m

def quad_dist(beta, beta_star, mask):
    # d_Q(x,y) := sqrt( |(x-y)^T M (x-y)| )  with diagonal M=mask
    d = beta - beta_star
    q = float(np.sum(mask * (d*d)))
    return math.sqrt(abs(q))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hepdata_lyapunov_test_out/per_label_timeseries.csv")
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--out", default="fejer_report.csv")
    ap.add_argument("--use_quadratic", action="store_true",
                    help="Also compute quadratic Fejer using --signature on β.")
    ap.add_argument("--signature", default="3,1",
                    help='For β dimension=5, give e.g. "++++-" or "4,1" or "+,+,+,+,-".')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # expected columns (based on your compare output): label, iter, b0..b4, chi2_dof, alpha, etc.
    beta_cols = [c for c in df.columns if c.startswith("b") and c[1:].isdigit()]
    if not beta_cols:
        raise RuntimeError("No beta columns found (expected b0..b4).")

    df = df.sort_values(["label","iter"]).reset_index(drop=True)

    rows = []
    for label, g in df.groupby("label"):
        g = g.sort_values("iter")
        B = g[beta_cols].to_numpy(dtype=float)
        iters = g["iter"].to_numpy(dtype=int)

        beta_star = B[-1]
        # L2 Fejer
        dL2 = np.linalg.norm(B - beta_star[None,:], axis=1)
        frac_L2, max_viol_L2, where_L2 = monotone_stats(dL2, eps=args.eps)

        # L1 Fejer (sometimes more stable)
        dL1 = np.linalg.norm(B - beta_star[None,:], ord=1, axis=1)
        frac_L1, max_viol_L1, where_L1 = monotone_stats(dL1, eps=args.eps)

        out = {
            "label": label,
            "T_iter": int(iters[-1]),
            "beta_dim": int(B.shape[1]),
            "fejer_frac_L2": frac_L2,
            "fejer_max_violation_L2": max_viol_L2,
            "fejer_violation_iter_L2": int(iters[where_L2]) if len(iters) else 0,
            "fejer_frac_L1": frac_L1,
            "fejer_max_violation_L1": max_viol_L1,
            "fejer_violation_iter_L1": int(iters[where_L1]) if len(iters) else 0,
        }

        if args.use_quadratic:
            mask = parse_signature(args.signature, dim=B.shape[1])
            dQ = np.array([quad_dist(B[i], beta_star, mask) for i in range(B.shape[0])], dtype=float)
            frac_Q, max_viol_Q, where_Q = monotone_stats(dQ, eps=args.eps)
            out.update({
                "signature": args.signature,
                "fejer_frac_Q": frac_Q,
                "fejer_max_violation_Q": max_viol_Q,
                "fejer_violation_iter_Q": int(iters[where_Q]),
            })

        rows.append(out)

    rep = pd.DataFrame(rows).sort_values("label")
    rep.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out}")
    print(rep.to_string(index=False))

if __name__ == "__main__":
    main()
