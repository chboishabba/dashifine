#!/usr/bin/env python3
"""
Seam-certificates for DASHI physics closure (finite empirical checks).

Inputs:
  - per_label_timeseries.csv: rows = (label, iter, b0..b4, chi2_dof, alpha, ...)
  - per_label_energy_report.csv (optional): summary / cached energies

Outputs:
  - fejer_set_report.csv: Fejér-to-fixed-set for many y in Fix(P)
  - transinv_report.csv: translation invariance check for chosen metric
  - mdl_descent_report.csv: exact monotone descent check
  - closestpoint_report.csv: empirical prox/closest-point check
  - cone_strict_report.csv: cone monotonicity with strict-premise filter
  - offenders.csv: which labels violate which seams
"""

import argparse, os, math
import numpy as np
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------

def beta_cols(df):
    return [c for c in df.columns if c.startswith("b") and c[1:].isdigit()]

def group_by_label(df):
    for label, g in df.groupby("label"):
        gg = g.sort_values("iter").reset_index(drop=True)
        yield label, gg

def l1(x): return float(np.sum(np.abs(x)))
def l2(x): return float(np.sqrt(np.sum(x*x)))

def mdllike_energy(beta, eps=1e-12):
    """
    Simple MDL proxy:
      model_bits ~ log(1 + ||beta||_1)
      resid_bits ~ log(1 + ||beta_tail||_1)  (or anything you like)
    """
    b = np.asarray(beta, dtype=float)
    model = math.log(1.0 + np.sum(np.abs(b)) + eps)
    tail  = math.log(1.0 + np.sum(np.abs(b[2:])) + eps)  # example: tail = b2.. (edit as desired)
    return model + tail

def make_fixP_sampler(dim, rng):
    """
    Fix(P) in your current beta collapse is empirically near 0-vector.
    But we sample a small neighborhood so 'random y in Fix(P)' isn't degenerate.

    If you later have an explicit Fix(P) parametrization, replace this.
    """
    def sample(n, scale=1e-3):
        return rng.normal(0.0, scale, size=(n, dim))
    return sample

# ----------------------------
# Seams
# ----------------------------

def seam_mdl_descent(label_g, energy_fn):
    betas = label_g[beta_cols(label_g)].to_numpy(dtype=float)
    E = np.array([energy_fn(b) for b in betas], dtype=float)
    diffs = E[1:] - E[:-1]
    violations = np.where(diffs > 0)[0]  # index t where E_{t+1} > E_t
    worst = float(diffs.max()) if diffs.size else 0.0
    return {
        "mdl_monotone_all": int(len(violations) == 0),
        "mdl_monotone_frac": float(np.mean(diffs <= 0)) if diffs.size else 1.0,
        "n_violations": int(len(violations)),
        "worst_increase": worst,
        "first_violation_iter": int(violations[0]) if len(violations) else -1,
        "last_iter": int(label_g["iter"].max()),
    }

def seam_fejer_to_fixed_set(label_g, dist_fn, n_y=256, seed=0):
    """
    Fejér-to-fixed-set certificate:
      for random y in Fix(P), verify d(Px_t, y) <= d(x_t, y) for all t.
    """
    rng = np.random.default_rng(seed)
    betas = label_g[beta_cols(label_g)].to_numpy(dtype=float)
    dim = betas.shape[1]
    sample_y = make_fixP_sampler(dim, rng)

    # Define Px as "next iterate" proxy (the empirical P is the projector you apply each step).
    # If you have explicit P(x), replace this with P(x_t).
    Px = betas[1:]
    x  = betas[:-1]

    ys = sample_y(n_y)
    # For each y, check Fejér over all t:
    ok_per_y = []
    worst_violation = 0.0
    worst_at = (-1, -1)  # (y_idx, t_idx)

    for yi, y in enumerate(ys):
        dx = np.array([dist_fn(xi - y) for xi in x], dtype=float)
        dP = np.array([dist_fn(pxi - y) for pxi in Px], dtype=float)
        # Fejér condition: dP <= dx
        viol = dP - dx
        ok = np.all(viol <= 0)
        ok_per_y.append(ok)
        vmax = float(viol.max()) if viol.size else 0.0
        if vmax > worst_violation:
            worst_violation = vmax
            t_idx = int(np.argmax(viol))
            worst_at = (yi, t_idx)

    return {
        "fejer_set_all": int(all(ok_per_y)),
        "fejer_set_frac": float(np.mean(ok_per_y)),
        "worst_violation": float(worst_violation),
        "worst_y_idx": int(worst_at[0]),
        "worst_t_idx": int(worst_at[1]),
        "n_y": int(n_y),
        "T_iter": int(label_g["iter"].max()),
    }

def seam_translation_invariance(label_g, dist_fn, n_trials=2000, seed=0):
    """
    Translation invariance: d(x+z,y+z)=d(x,y)
    In beta-space, this should hold for L1/L2; for MDL-proxy distance, it may not.
    """
    rng = np.random.default_rng(seed)
    betas = label_g[beta_cols(label_g)].to_numpy(dtype=float)
    dim = betas.shape[1]
    if len(betas) < 2:
        return {"transinv_frac": 1.0, "worst_abs_err": 0.0, "n_trials": 0}

    worst = 0.0
    ok = 0
    for _ in range(n_trials):
        i, j = rng.integers(0, len(betas), size=2)
        x, y = betas[i], betas[j]
        z = rng.normal(0.0, 1.0, size=(dim,))
        d0 = dist_fn(x - y)
        d1 = dist_fn((x + z) - (y + z))
        err = abs(d1 - d0)
        worst = max(worst, float(err))
        if err < 1e-9:
            ok += 1
    return {"transinv_frac": ok / n_trials, "worst_abs_err": float(worst), "n_trials": int(n_trials)}

def seam_closest_point(label_g, energy_dist_fn, n_y=256, seed=0):
    """
    ClosestPoint empirical:
      E(x, Px) <= E(x, y) for random y in Fix(P), across many x.
    Here energy_dist_fn is a distance-like function on (x,y), e.g. ||x-y||_1.
    """
    rng = np.random.default_rng(seed)
    betas = label_g[beta_cols(label_g)].to_numpy(dtype=float)
    dim = betas.shape[1]
    ys = make_fixP_sampler(dim, rng)(n_y)

    Px = betas[1:]
    x  = betas[:-1]

    total = 0
    ok = 0
    worst = 0.0
    worst_at = (-1, -1, -1)  # (x_idx, y_idx, t)

    for t in range(len(x)):
        xt = x[t]
        pxt = Px[t]
        ExP = energy_dist_fn(xt, pxt)
        for yi, y in enumerate(ys):
            Exy = energy_dist_fn(xt, y)
            total += 1
            diff = ExP - Exy
            if diff <= 0:
                ok += 1
            else:
                if diff > worst:
                    worst = float(diff)
                    worst_at = (t, yi, int(label_g.loc[t, "iter"]))
    return {
        "closest_frac": ok / total if total else 1.0,
        "closest_all": int(ok == total),
        "worst_margin": float(worst),
        "worst_t": int(worst_at[0]),
        "worst_y": int(worst_at[1]),
        "worst_iter": int(worst_at[2]),
        "n_pairs": int(total),
    }

def seam_cone_strict(label_g, dnatfine_col="dNatFine", eps=0.0):
    """
    Cone monotonicity strict-premise filter:
      only evaluate pairs where dNatFine_t > 0 (or > eps).

    This assumes you already computed/attached dNatFine per iter.
    If not present, this seam reports 'skipped'.
    """
    if dnatfine_col not in label_g.columns:
        return {"status": "skipped_no_dNatFine"}

    dn = label_g[dnatfine_col].to_numpy(dtype=float)
    # premise is about transitions t -> t+1
    dn_t = dn[:-1]
    mask = dn_t > eps
    if not np.any(mask):
        return {"status": "skipped_all_premise_false", "eps": float(eps)}

    # Placeholder: if you already have a cone measure per step, plug it here.
    # For now we use "dn decreases" as a proxy (replace with your real cone condition).
    dn_next = dn[1:]
    viol = (dn_next[mask] > dn_t[mask])
    return {
        "status": "ok",
        "eps": float(eps),
        "n_checked": int(mask.sum()),
        "violations": int(viol.sum()),
        "viol_frac": float(np.mean(viol)) if mask.sum() else 0.0,
    }

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True, help="per_label_timeseries.csv")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--transinv-trials", type=int, default=2000)
    ap.add_argument("--cone-eps", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.timeseries)

    # Choose distances / energies
    dist_L1 = lambda v: l1(v)
    dist_L2 = lambda v: l2(v)
    # energy distance for ClosestPoint:
    E_L1 = lambda x,y: l1(x-y)
    E_L2 = lambda x,y: l2(x-y)
    # MDL proxy energy (scalar) descent:
    energy_fn = mdllike_energy

    rows_mdl = []
    rows_fejer_set = []
    rows_trans = []
    rows_close = []
    rows_cone = []
    offenders = []

    for label, g in group_by_label(df):
        # MDL descent
        r_mdl = seam_mdl_descent(g, energy_fn)
        rows_mdl.append({"label": label, **r_mdl})
        if not r_mdl["mdl_monotone_all"]:
            offenders.append({"label": label, "seam": "MDLDescent", "detail": r_mdl})

        # Fejér-to-fixed-set (L1 and L2)
        r_f1 = seam_fejer_to_fixed_set(g, dist_L1, n_y=args.ny, seed=args.seed)
        r_f2 = seam_fejer_to_fixed_set(g, dist_L2, n_y=args.ny, seed=args.seed)
        rows_fejer_set.append({"label": label, "metric": "L1", **r_f1})
        rows_fejer_set.append({"label": label, "metric": "L2", **r_f2})
        if not r_f1["fejer_set_all"]:
            offenders.append({"label": label, "seam": "FejerSet_L1", "detail": r_f1})
        if not r_f2["fejer_set_all"]:
            offenders.append({"label": label, "seam": "FejerSet_L2", "detail": r_f2})

        # Translation invariance (L1 and L2)
        r_t1 = seam_translation_invariance(g, dist_L1, n_trials=args.transinv_trials, seed=args.seed)
        r_t2 = seam_translation_invariance(g, dist_L2, n_trials=args.transinv_trials, seed=args.seed)
        rows_trans.append({"label": label, "metric": "L1", **r_t1})
        rows_trans.append({"label": label, "metric": "L2", **r_t2})

        # ClosestPoint (L1 and L2)
        r_c1 = seam_closest_point(g, E_L1, n_y=args.ny, seed=args.seed)
        r_c2 = seam_closest_point(g, E_L2, n_y=args.ny, seed=args.seed)
        rows_close.append({"label": label, "metric": "L1", **r_c1})
        rows_close.append({"label": label, "metric": "L2", **r_c2})
        if not r_c1["closest_all"]:
            offenders.append({"label": label, "seam": "ClosestPoint_L1", "detail": r_c1})

        # Cone strict premise filter (if dNatFine present)
        r_cone = seam_cone_strict(g, eps=args.cone_eps)
        rows_cone.append({"label": label, **r_cone})

    pd.DataFrame(rows_mdl).to_csv(os.path.join(args.out, "mdl_descent_report.csv"), index=False)
    pd.DataFrame(rows_fejer_set).to_csv(os.path.join(args.out, "fejer_set_report.csv"), index=False)
    pd.DataFrame(rows_trans).to_csv(os.path.join(args.out, "transinv_report.csv"), index=False)
    pd.DataFrame(rows_close).to_csv(os.path.join(args.out, "closestpoint_report.csv"), index=False)
    pd.DataFrame(rows_cone).to_csv(os.path.join(args.out, "cone_strict_report.csv"), index=False)
    pd.DataFrame(offenders).to_csv(os.path.join(args.out, "offenders.csv"), index=False)

    print("[ok] wrote seam reports to:", args.out)

if __name__ == "__main__":
    main()
