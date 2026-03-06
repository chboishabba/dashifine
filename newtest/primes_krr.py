import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from dashifine.kernels import dashifine_kernel


def periodic_rbf_kernel(X1, X2, lengthscale=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    diff = np.sin(diff * np.pi / (2 * np.pi)) * 2 * np.pi
    d2 = np.sum(diff**2, axis=-1)
    return np.exp(-0.5 * d2 / (lengthscale**2))


def krr_fit_predict(Xtr, ytr, Xte, kernel_fn, lam):
    K = kernel_fn(Xtr, Xtr)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), ytr)
    return kernel_fn(Xte, Xtr) @ alpha


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def v_p(n, p):
    count = 0
    while n % p == 0 and n > 0:
        n //= p
        count += 1
    return count


def residue_embedding(n_vals, m1, m2):
    a1 = 2 * np.pi * (n_vals % m1) / m1
    a2 = 2 * np.pi * (n_vals % m2) / m2
    return np.stack([a1, a2], axis=1).astype(np.float32)


def save_divisibility_plot(ps, dash_mse, prbf_mse, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ps))
    ax.bar(x - 0.15, dash_mse, width=0.3, label="Dashifine")
    ax.bar(x + 0.15, prbf_mse, width=0.3, label="pRBF")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("Prime p")
    ax.set_ylabel("MSE")
    ax.set_title("Divisibility task MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_vp_plot(n_vals, y_true, preds, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_vals, y_true, label="True", lw=1.5)
    ax.plot(n_vals, preds["dashifine"], label="Dashifine", lw=1.2)
    ax.plot(n_vals, preds["prbf"], label="pRBF", lw=1.2)
    ax.set_xlabel("n")
    ax.set_ylabel("v_p(n) (normalized)")
    ax.set_title("p-adic valuation task")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_indicator_plot(levels, dash_mse, prbf_mse, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(levels))
    ax.bar(x - 0.15, dash_mse, width=0.3, label="Dashifine")
    ax.bar(x + 0.15, prbf_mse, width=0.3, label="pRBF")
    ax.set_xticks(x)
    ax.set_xticklabels([str(level) for level in levels])
    ax.set_xlabel("k (indicator 1[p^k | n])")
    ax.set_ylabel("MSE")
    ax.set_title("Valuation indicator MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/primes_krr")
    parser.add_argument("--n_max", type=int, default=512)
    parser.add_argument("--m1", type=int, default=8)
    parser.add_argument("--m2", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--lengthscale", type=float, default=0.7)
    parser.add_argument("--p_list", type=str, default="2,3,5,7")
    parser.add_argument("--vp_prime", type=int, default=2)
    parser.add_argument("--vp_cap", type=int, default=6)
    parser.add_argument("--indicator_max_power", type=int, default=4)
    args = parser.parse_args()

    root_dir = Path(args.output_dir)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root_dir / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_vals = np.arange(1, args.n_max + 1)
    X = residue_embedding(n_vals, args.m1, args.m2)

    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    Xtr, Xte = X[tr_idx], X[te_idx]
    n_tr, n_te = n_vals[tr_idx], n_vals[te_idx]

    dash_kernel = lambda A, B: dashifine_kernel(
        A, B, k_max=4, temperature=args.temperature
    )
    prbf_kernel = lambda A, B: periodic_rbf_kernel(
        A, B, lengthscale=args.lengthscale
    )

    primes = [int(p) for p in args.p_list.split(",") if p.strip()]
    dash_mse = []
    prbf_mse = []
    div_lines = []
    for p in primes:
        y = (n_vals % p == 0).astype(np.float32)
        ytr, yte = y[tr_idx], y[te_idx]
        pred_dash = krr_fit_predict(Xtr, ytr, Xte, dash_kernel, args.lam)
        pred_prbf = krr_fit_predict(Xtr, ytr, Xte, prbf_kernel, args.lam)
        mse_dash = mse(pred_dash, yte)
        mse_prbf = mse(pred_prbf, yte)
        dash_mse.append(mse_dash)
        prbf_mse.append(mse_prbf)
        div_lines.append(f"div_p={p} mse_dash={mse_dash:.6f} mse_prbf={mse_prbf:.6f}")

    save_divisibility_plot(primes, dash_mse, prbf_mse, out_dir / "divisibility_mse.png")

    vp_vals = np.array([min(v_p(n, args.vp_prime), args.vp_cap) for n in n_vals])
    vp_norm = vp_vals.astype(np.float32) / float(args.vp_cap)
    vp_tr, vp_te = vp_norm[tr_idx], vp_norm[te_idx]
    pred_dash_vp = krr_fit_predict(Xtr, vp_tr, Xte, dash_kernel, args.lam)
    pred_prbf_vp = krr_fit_predict(Xtr, vp_tr, Xte, prbf_kernel, args.lam)
    vp_mse_dash = mse(pred_dash_vp, vp_te)
    vp_mse_prbf = mse(pred_prbf_vp, vp_te)

    preds_full = {
        "dashifine": krr_fit_predict(Xtr, vp_tr, X, dash_kernel, args.lam),
        "prbf": krr_fit_predict(Xtr, vp_tr, X, prbf_kernel, args.lam),
    }
    save_vp_plot(n_vals, vp_norm, preds_full, out_dir / "vp_prediction.png")

    indicator_levels = []
    dash_ind_mse = []
    prbf_ind_mse = []
    ind_lines = []
    for k in range(1, args.indicator_max_power + 1):
        if args.vp_prime ** k > args.n_max:
            continue
        indicator_levels.append(k)
        y_ind = (n_vals % (args.vp_prime ** k) == 0).astype(np.float32)
        ytr_ind, yte_ind = y_ind[tr_idx], y_ind[te_idx]
        pred_dash_ind = krr_fit_predict(Xtr, ytr_ind, Xte, dash_kernel, args.lam)
        pred_prbf_ind = krr_fit_predict(Xtr, ytr_ind, Xte, prbf_kernel, args.lam)
        mse_dash_ind = mse(pred_dash_ind, yte_ind)
        mse_prbf_ind = mse(pred_prbf_ind, yte_ind)
        dash_ind_mse.append(mse_dash_ind)
        prbf_ind_mse.append(mse_prbf_ind)
        ind_lines.append(
            f"indicator_k={k} mse_dash={mse_dash_ind:.6f} mse_prbf={mse_prbf_ind:.6f}"
        )

    if indicator_levels:
        save_indicator_plot(
            indicator_levels,
            dash_ind_mse,
            prbf_ind_mse,
            out_dir / "indicator_mse.png",
        )

    log_lines = [
        "Primes/divisibility KRR benchmark",
        f"output_dir={out_dir}",
        f"n_max={args.n_max} m1={args.m1} m2={args.m2} seed={args.seed}",
        f"lam={args.lam} temperature={args.temperature} lengthscale={args.lengthscale}",
        f"p_list={primes}",
        *div_lines,
        f"vp_prime={args.vp_prime} vp_cap={args.vp_cap}",
        f"vp_mse_dash={vp_mse_dash:.6f} vp_mse_prbf={vp_mse_prbf:.6f}",
        f"indicator_max_power={args.indicator_max_power}",
        *ind_lines,
        "outputs=divisibility_mse.png vp_prediction.png indicator_mse.png",
    ]
    summary_path = out_dir / "run_summary.txt"
    summary_path.write_text("\n".join(log_lines))

    for line in log_lines:
        print(line)


if __name__ == "__main__":
    main()
