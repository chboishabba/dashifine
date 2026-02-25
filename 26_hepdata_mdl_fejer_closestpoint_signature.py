#!/usr/bin/env python3
import argparse
import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set

import numpy as np
import pandas as pd


BETA_COLS = ["b0", "b1", "b2", "b3", "b4"]


# -----------------------------
# Projection P (closest-point target you can certify today)
# -----------------------------
def P_mask_b1_b3(x: np.ndarray) -> np.ndarray:
    """P(b0,b1,b2,b3,b4) = (b0,0,b2,0,b4). Fix(P) is the subspace b1=b3=0."""
    y = x.copy()
    y[1] = 0.0
    y[3] = 0.0
    return y


# -----------------------------
# Distances / energies
# -----------------------------
def d_L2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=2))


def d_L1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=1))


@dataclass(frozen=True)
class MaskedQuadratic:
    """
    Q(x) = sum_i s_i x_i^2 where s_i in {+1, -1, 0}
    signature reported as (n_pos, n_neg, n_zero).
    """
    s: np.ndarray  # shape (dim,), entries in {-1,0,+1}

    def signature(self) -> Tuple[int, int, int]:
        n_pos = int(np.sum(self.s > 0))
        n_neg = int(np.sum(self.s < 0))
        n_zero = int(np.sum(self.s == 0))
        return (n_pos, n_neg, n_zero)

    def Q(self, x: np.ndarray) -> float:
        return float(np.sum(self.s * (x ** 2)))

    def dQ(self, a: np.ndarray, b: np.ndarray) -> float:
        """Indefinite 'distance' proxy: sqrt(|Q(a-b)|)."""
        v = a - b
        return float(math.sqrt(abs(self.Q(v))))


# -----------------------------
# Loading
# -----------------------------
def load_timeseries(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"label", "iter"} | set(BETA_COLS)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"timeseries missing columns: {sorted(missing)}")

    # MDL proxy column name: prefer E_MDL_proxy if present
    if "E_MDL_proxy" not in df.columns:
        raise ValueError("timeseries missing E_MDL_proxy (expected your MDL proxy column).")

    df = df.copy()
    df["iter"] = df["iter"].astype(int)
    return df.sort_values(["label", "iter"]).reset_index(drop=True)


# -----------------------------
# 1) Exact MDL descent (Fejér-to-final under MDL proxy)
# -----------------------------
def check_exact_mdl_descent(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby("label", sort=False):
        e = g["E_MDL_proxy"].to_numpy(float)
        it = g["iter"].to_numpy(int)

        diffs = np.diff(e)
        viol_idx = np.where(diffs > eps)[0]  # e[t+1] - e[t] > eps
        rows.append({
            "label": label,
            "T_iter": int(it[-1]),
            "n_steps": int(len(e) - 1),
            "eps": eps,
            "mdl_monotone_all": bool(len(viol_idx) == 0),
            "n_violations": int(len(viol_idx)),
            "worst_increase": float(diffs[viol_idx].max()) if len(viol_idx) else 0.0,
            "worst_violation_iter": int(it[viol_idx[diffs[viol_idx].argmax()]]) if len(viol_idx) else -1,
        })
    return pd.DataFrame(rows)


# -----------------------------
# 2) ClosestPoint empirical test (projection is proximal)
# -----------------------------
def closest_point_test(
    df: pd.DataFrame,
    proj_fn,
    dist_fn,
    n_y: int,
    noise_scale: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    free_axes = [0, 2, 4]  # b0,b2,b4 are free within Fix(P) for this P
    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        viol_total = 0
        trials_total = 0
        worst_gap = 0.0  # d(x,Px) - d(x,y_best)
        worst_at_iter = -1

        for _, r in g.iterrows():
            x = r[BETA_COLS].to_numpy(float)
            Px = proj_fn(x)
            dPx = dist_fn(x, Px)

            best_d = float("inf")
            # sample random y in Fix(P): y = Px + noise on free axes
            for _k in range(n_y):
                y = Px.copy()
                for ax in free_axes:
                    y[ax] += rng.normal(0.0, noise_scale)
                dy = dist_fn(x, y)
                if dy < best_d:
                    best_d = dy

            # violation if some sampled y beats Px by more than tiny tolerance
            gap = dPx - best_d
            if gap > 1e-12:
                viol_total += 1
                worst_gap = max(worst_gap, gap)
                worst_at_iter = int(r["iter"])

            trials_total += 1

        rows.append({
            "label": label,
            "dist": getattr(dist_fn, "__name__", "dist"),
            "n_points": trials_total,
            "n_y": n_y,
            "noise_scale": noise_scale,
            "viol_rate": viol_total / max(1, trials_total),
            "worst_gap": worst_gap,
            "worst_violation_iter": worst_at_iter,
        })

    return pd.DataFrame(rows)


# -----------------------------
# 3) Parallelogram residual monitor (numeric)
# -----------------------------
def parallelogram_residual(df: pd.DataFrame, dist2_fn, seed: int, n_pairs: int) -> pd.DataFrame:
    """
    For a norm induced by an inner product: ||x+y||^2 + ||x-y||^2 = 2||x||^2 + 2||y||^2.
    We measure residual:
      R = (||x+y||^2 + ||x-y||^2) - 2||x||^2 - 2||y||^2
    Here dist2_fn should return a squared norm proxy of a vector.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for label, g in df.groupby("label", sort=False):
        X = g.sort_values("iter")[BETA_COLS].to_numpy(float)
        it = g.sort_values("iter")["iter"].to_numpy(int)

        # compute residual per iter by sampling pairs around that iter (use x_t and x_rand)
        # This is a monitor, not a proof: you want it to shrink as you approach IR.
        for t in range(len(X)):
            x = X[t]
            res_vals = []
            for _ in range(n_pairs):
                y = X[rng.integers(0, len(X))]
                lhs = dist2_fn(x + y) + dist2_fn(x - y)
                rhs = 2.0 * dist2_fn(x) + 2.0 * dist2_fn(y)
                res_vals.append(lhs - rhs)
            rows.append({
                "label": label,
                "iter": int(it[t]),
                "parallelogram_res_mean": float(np.mean(res_vals)),
                "parallelogram_res_absmean": float(np.mean(np.abs(res_vals))),
                "parallelogram_res_maxabs": float(np.max(np.abs(res_vals))),
            })
    return pd.DataFrame(rows)


# -----------------------------
# 4) MaskedQuadratic preservation/descent + cone check (empirical)
# -----------------------------
def masked_quadratic_checks(
    df: pd.DataFrame,
    mq: MaskedQuadratic,
    eps: float,
    cone_eps: float,
) -> Dict[str, pd.DataFrame]:
    # Q descent: Q(x_{t+1}) <= Q(x_t)
    q_rows = []
    cone_rows = []

    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("iter")
        X = g[BETA_COLS].to_numpy(float)
        it = g["iter"].to_numpy(int)
        Qs = np.array([mq.Q(x) for x in X], dtype=float)

        diffs = np.diff(Qs)
        viol = np.where(diffs > eps)[0]

        q_rows.append({
            "label": label,
            "signature": str(mq.signature()),
            "eps": eps,
            "Q_monotone_all": bool(len(viol) == 0),
            "Q_monotone_frac": float(np.mean(diffs <= eps)) if len(diffs) else 1.0,
            "n_violations": int(len(viol)),
            "worst_increase": float(diffs[viol].max()) if len(viol) else 0.0,
            "worst_violation_iter": int(it[viol[diffs[viol].argmax()]]) if len(viol) else -1,
        })

        # Cone preservation: if Q(x_t) >= -cone_eps, check Q(x_{t+1}) >= -cone_eps
        total = 0
        ok = 0
        worst = 0.0
        worst_it = -1
        for t in range(len(Qs) - 1):
            if Qs[t] >= -cone_eps:
                total += 1
                if Qs[t + 1] >= -cone_eps:
                    ok += 1
                else:
                    # how far below the cone did we go?
                    drop = (-cone_eps) - Qs[t + 1]
                    if drop > worst:
                        worst = float(drop)
                        worst_it = int(it[t])
        cone_rows.append({
            "label": label,
            "signature": str(mq.signature()),
            "cone_eps": cone_eps,
            "cone_pres_frac": (ok / total) if total else float("nan"),
            "cone_total": int(total),
            "cone_violations": int(total - ok),
            "cone_worst_below": worst,
            "cone_worst_at_iter": worst_it,
        })

    return {
        "Q_descent": pd.DataFrame(q_rows),
        "cone_preservation": pd.DataFrame(cone_rows),
    }


# -----------------------------
# 5) Signature elimination screen (heuristic)
# -----------------------------
def enumerate_signatures(dim: int, p: int, q: int, z: int) -> Iterable[np.ndarray]:
    """Return all sign vectors s with exactly p +1s, q -1s, z 0s."""
    assert p + q + z == dim
    idx = list(range(dim))
    for pos in itertools.combinations(idx, p):
        remaining = [i for i in idx if i not in pos]
        for neg in itertools.combinations(remaining, q):
            s = np.zeros(dim, dtype=float)
            s[list(pos)] = 1.0
            s[list(neg)] = -1.0
            yield s


def signature_screen(df: pd.DataFrame, patterns: List[Tuple[int, int, int]], eps: float, cone_eps: float) -> pd.DataFrame:
    rows = []
    for (p, q, z) in patterns:
        for s in enumerate_signatures(dim=5, p=p, q=q, z=z):
            mq = MaskedQuadratic(s=s)
            checks = masked_quadratic_checks(df, mq, eps=eps, cone_eps=cone_eps)
            qtab = checks["Q_descent"]
            ctab = checks["cone_preservation"]

            # aggregate score: prefer high Q_monotone_frac and high cone_pres_frac
            qscore = float(qtab["Q_monotone_frac"].mean())
            # cone_pres_frac can be nan if cone never entered; treat nan as 0 here
            cone_vals = ctab["cone_pres_frac"].to_numpy(float)
            cone_vals = np.nan_to_num(cone_vals, nan=0.0)
            cscore = float(np.mean(cone_vals))

            rows.append({
                "signature": str((p, q, z)),
                "s_vector": ",".join(str(int(v)) for v in s.tolist()),
                "Q_monotone_frac_mean": qscore,
                "cone_pres_frac_mean": cscore,
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["Q_monotone_frac_mean", "cone_pres_frac_mean"], ascending=False).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True, help="per_label_timeseries.csv (has betas + E_MDL_proxy)")
    ap.add_argument("--out-prefix", default="hepdata_mdl_tests", help="output prefix (csv files)")
    ap.add_argument("--eps", type=float, default=0.0, help="epsilon for monotone checks")
    ap.add_argument("--cone-eps", type=float, default=1e-9, help="epsilon for cone membership Q>=-cone_eps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--closest-n-y", type=int, default=500, help="random y samples per x for closest-point test")
    ap.add_argument("--closest-noise", type=float, default=0.5, help="noise scale for random y in Fix(P)")
    ap.add_argument("--par-n-pairs", type=int, default=100, help="pairs per iter for parallelogram residual monitor")
    ap.add_argument("--screen", action="store_true", help="run signature screening (can be slow)")
    args = ap.parse_args()

    df = load_timeseries(args.timeseries)

    # 1) Exact MDL descent
    mdl = check_exact_mdl_descent(df, eps=args.eps)

    # 2) Closest point tests (L2, L1, and one masked quadratic default)
    cp_L2 = closest_point_test(df, P_mask_b1_b3, d_L2, args.closest_n_y, args.closest_noise, args.seed)
    cp_L1 = closest_point_test(df, P_mask_b1_b3, d_L1, args.closest_n_y, args.closest_noise, args.seed)

    # Default masked quadratic: treat (b0,b2,b4) as +, (b1,b3) as 0 (pure gauge axes)
    s_default = np.array([1, 0, 1, 0, 1], dtype=float)
    mq_default = MaskedQuadratic(s=s_default)
    cp_MQ = closest_point_test(
        df,
        P_mask_b1_b3,
        lambda a, b: mq_default.dQ(a, b),
        args.closest_n_y,
        args.closest_noise,
        args.seed,
    )

    # 3) Parallelogram residual monitor for the *positive* part of default MQ
    def norm2_pos(x: np.ndarray) -> float:
        # squared norm using only + axes of mq_default (b0,b2,b4)
        return float(np.sum((x[[0, 2, 4]] ** 2)))

    par = parallelogram_residual(df, dist2_fn=norm2_pos, seed=args.seed, n_pairs=args.par_n_pairs)

    # 4) MaskedQuadratic checks for a few candidate "physics-like" patterns in beta-space
    # (Heuristic; you can change these)
    candidates = [
        ("all_pos_(5,0,0)", np.array([1, 1, 1, 1, 1], float)),
        ("gauge_zero_(3,0,2)", np.array([1, 0, 1, 0, 1], float)),
        ("one_neg_(4,1,0)_neg_b2", np.array([1, 1, -1, 1, 1], float)),
        ("one_neg_(4,1,0)_neg_b0", np.array([-1, 1, 1, 1, 1], float)),
    ]
    mq_tabs = []
    cone_tabs = []
    for name, s in candidates:
        mq = MaskedQuadratic(s=s)
        checks = masked_quadratic_checks(df, mq, eps=args.eps, cone_eps=args.cone_eps)
        qd = checks["Q_descent"].copy()
        qd.insert(1, "name", name)
        cpv = checks["cone_preservation"].copy()
        cpv.insert(1, "name", name)
        mq_tabs.append(qd)
        cone_tabs.append(cpv)
    mq_Q = pd.concat(mq_tabs, ignore_index=True)
    mq_cone = pd.concat(cone_tabs, ignore_index=True)

    # 5) Optional signature screen
    screen_df = None
    if args.screen:
        patterns = [
            (4, 1, 0),  # “Lorentz-like” in dim 5 (heuristic)
            (3, 1, 1),
            (3, 2, 0),
            (5, 0, 0),
            (4, 0, 1),
        ]
        screen_df = signature_screen(df, patterns=patterns, eps=args.eps, cone_eps=args.cone_eps)

    # Write outputs
    out0 = f"{args.out_prefix}_mdl_descent.csv"
    out1 = f"{args.out_prefix}_closestpoint_L2.csv"
    out2 = f"{args.out_prefix}_closestpoint_L1.csv"
    out3 = f"{args.out_prefix}_closestpoint_MQ.csv"
    out4 = f"{args.out_prefix}_parallelogram_monitor.csv"
    out5 = f"{args.out_prefix}_maskedQ_descent.csv"
    out6 = f"{args.out_prefix}_maskedQ_cone.csv"
    mdl.to_csv(out0, index=False)
    cp_L2.to_csv(out1, index=False)
    cp_L1.to_csv(out2, index=False)
    cp_MQ.to_csv(out3, index=False)
    par.to_csv(out4, index=False)
    mq_Q.to_csv(out5, index=False)
    mq_cone.to_csv(out6, index=False)

    print(f"[ok] wrote {out0}")
    print(f"[ok] wrote {out1}")
    print(f"[ok] wrote {out2}")
    print(f"[ok] wrote {out3}")
    print(f"[ok] wrote {out4}")
    print(f"[ok] wrote {out5}")
    print(f"[ok] wrote {out6}")

    if screen_df is not None:
        out7 = f"{args.out_prefix}_signature_screen.csv"
        screen_df.to_csv(out7, index=False)
        print(f"[ok] wrote {out7}")


if __name__ == "__main__":
    main()
