import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from dashifine.kernels import dashifine_kernel


def rbf_kernel(X1, X2, lengthscale=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
    d2 = X1_sq + X2_sq - 2 * (X1 @ X2.T)
    return np.exp(-0.5 * d2 / (lengthscale**2))


def periodic_rbf_kernel(X1, X2, lengthscale=1.0, period=2 * np.pi):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    diff = np.sin(diff * np.pi / period) * (period / np.pi)
    d2 = np.sum(diff**2, axis=-1)
    return np.exp(-0.5 * d2 / (lengthscale**2))


def mixed_kernel(kernel_xy, kernel_state):
    def _kernel(X1, X2):
        K_xy = kernel_xy(X1[:, :2], X2[:, :2])
        K_state = kernel_state(X1[:, 2:], X2[:, 2:])
        return K_xy * K_state

    return _kernel


def kernel_spectrum(X, kernel_fn, lam=1e-12):
    K = kernel_fn(X, X)
    K = K + lam * np.eye(len(X))
    eigvals = np.linalg.eigvalsh(K)
    return eigvals[::-1]


def gray_scott_step(u, v, du, dv, f, k, dt):
    lap_u = (
        np.roll(u, 1, 0)
        + np.roll(u, -1, 0)
        + np.roll(u, 1, 1)
        + np.roll(u, -1, 1)
        - 4.0 * u
    )
    lap_v = (
        np.roll(v, 1, 0)
        + np.roll(v, -1, 0)
        + np.roll(v, 1, 1)
        + np.roll(v, -1, 1)
        - 4.0 * v
    )
    uvv = u * v * v
    u_next = u + (du * lap_u - uvv + f * (1.0 - u)) * dt
    v_next = v + (dv * lap_v + uvv - (f + k) * v) * dt
    return u_next, v_next


def simulate_gray_scott(
    grid=32,
    steps=200,
    du=0.16,
    dv=0.08,
    f=0.06,
    k=0.062,
    dt=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    u = np.ones((grid, grid), dtype=np.float32)
    v = np.zeros((grid, grid), dtype=np.float32)

    r = grid // 8
    c0 = grid // 2
    u[c0 - r : c0 + r, c0 - r : c0 + r] = 0.0
    v[c0 - r : c0 + r, c0 - r : c0 + r] = 1.0

    u += 0.02 * rng.standard_normal(u.shape).astype(np.float32)
    v += 0.02 * rng.standard_normal(v.shape).astype(np.float32)

    frames = []
    for _ in range(steps):
        u, v = gray_scott_step(u, v, du, dv, f, k, dt)
        frames.append((u.copy(), v.copy()))
    return frames


def build_dataset(frames, sample_frames, rng, grid, per_frame):
    xs = np.linspace(-np.pi, np.pi, grid, endpoint=False, dtype=np.float32)
    ys = np.linspace(-np.pi, np.pi, grid, endpoint=False, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([XX, YY], axis=-1).reshape(-1, 2)

    feats = []
    targets = []
    total_cells = grid * grid
    for t in sample_frames:
        u_t, v_t = frames[t]
        u_next, v_next = frames[t + 1]
        uv = np.stack([u_t, v_t], axis=-1).reshape(total_cells, 2)
        uv_next = np.stack([u_next, v_next], axis=-1).reshape(total_cells, 2)

        idx = rng.choice(total_cells, size=per_frame, replace=False)
        feats.append(np.concatenate([coords[idx], uv[idx]], axis=1))
        targets.append(uv_next[idx])

    return np.vstack(feats), np.vstack(targets)


def krr_fit_predict(Xtr, ytr, Xte, kernel_fn, lam):
    K = kernel_fn(Xtr, Xtr)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), ytr)
    return kernel_fn(Xte, Xtr) @ alpha


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def radial_profile(values, radii, bin_edges):
    bin_idx = np.digitize(radii, bin_edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    sums = np.bincount(bin_idx, weights=values, minlength=len(bin_edges) - 1)
    counts = np.bincount(bin_idx, minlength=len(bin_edges) - 1)
    return np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)


def save_spectrum_plot(eigs, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(eigs) + 1), eigs, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_field_comparison(grid, coords, u_true, v_true, preds, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    titles = ["True U", "Dashifine U", "pRBF U", "True V", "Dashifine V", "pRBF V"]
    data = [
        u_true,
        preds["dashifine"][:, 0].reshape(grid, grid),
        preds["prbf"][:, 0].reshape(grid, grid),
        v_true,
        preds["dashifine"][:, 1].reshape(grid, grid),
        preds["prbf"][:, 1].reshape(grid, grid),
    ]
    for ax, title, field in zip(axes.ravel(), titles, data):
        im = ax.imshow(field, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_rollout_snapshot(grid, u_true, v_true, preds, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    titles = [
        "True U",
        "Dashifine U",
        "pRBF U",
        "RBF U",
        "True V",
        "Dashifine V",
        "pRBF V",
        "RBF V",
    ]
    data = [
        u_true,
        preds["dashifine"][:, 0].reshape(grid, grid),
        preds["prbf"][:, 0].reshape(grid, grid),
        preds["rbf"][:, 0].reshape(grid, grid),
        v_true,
        preds["dashifine"][:, 1].reshape(grid, grid),
        preds["prbf"][:, 1].reshape(grid, grid),
        preds["rbf"][:, 1].reshape(grid, grid),
    ]
    for ax, title, field in zip(axes.ravel(), titles, data):
        im = ax.imshow(field, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/grayscott_krr")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--burn_in", type=int, default=40)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--per_frame", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--du", type=float, default=0.16)
    parser.add_argument("--dv", type=float, default=0.08)
    parser.add_argument("--f", type=float, default=0.06)
    parser.add_argument("--k", type=float, default=0.062)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--state_ell", type=float, default=0.5)
    parser.add_argument("--spatial_ell", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--rollout_snapshot_steps", type=str, default="1,5,10,20")
    parser.add_argument("--rollout_gif_steps", type=int, default=0)
    parser.add_argument("--rollout_gif_stride", type=int, default=1)
    parser.add_argument("--rollout_gif_fps", type=int, default=8)
    parser.add_argument("--keep_gif_frames", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.output_dir)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root_dir / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    frames = simulate_gray_scott(
        grid=args.grid,
        steps=args.steps,
        du=args.du,
        dv=args.dv,
        f=args.f,
        k=args.k,
        dt=args.dt,
        seed=args.seed,
    )

    available = np.arange(args.burn_in, args.steps - 1)
    sample_frames = rng.choice(available, size=args.frames, replace=False)
    X, y = build_dataset(frames, sample_frames, rng, args.grid, args.per_frame)

    n_train = int(0.8 * len(X))
    idx = rng.permutation(len(X))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    dashifine_xy = lambda A, B: dashifine_kernel(
        A, B, k_max=4, temperature=args.temperature
    )
    rbf_state = lambda A, B: rbf_kernel(A, B, lengthscale=args.state_ell)
    prbf_xy = lambda A, B: periodic_rbf_kernel(
        A, B, lengthscale=args.spatial_ell, period=2 * np.pi
    )
    rbf_xy = lambda A, B: rbf_kernel(A, B, lengthscale=args.spatial_ell)

    kernel_dash = mixed_kernel(dashifine_xy, rbf_state)
    kernel_prbf = mixed_kernel(prbf_xy, rbf_state)
    kernel_rbf = rbf_kernel

    eig_dash = kernel_spectrum(Xtr, kernel_dash)
    eig_prbf = kernel_spectrum(Xtr, kernel_prbf)
    eig_rbf = kernel_spectrum(Xtr, kernel_rbf)

    save_spectrum_plot(
        eig_dash[:50],
        out_dir / "spectrum_dashifine.png",
        "Dashifine spectrum (top 50)",
    )
    save_spectrum_plot(
        eig_prbf[:50],
        out_dir / "spectrum_prbf.png",
        "Periodic RBF spectrum (top 50)",
    )
    save_spectrum_plot(
        eig_rbf[:50],
        out_dir / "spectrum_rbf.png",
        "RBF spectrum (top 50)",
    )

    preds = {}
    preds["dashifine"] = krr_fit_predict(Xtr, ytr, Xte, kernel_dash, args.lam)
    preds["prbf"] = krr_fit_predict(Xtr, ytr, Xte, kernel_prbf, args.lam)
    preds["rbf"] = krr_fit_predict(Xtr, ytr, Xte, kernel_rbf, args.lam)

    mse_dash = mse(preds["dashifine"], yte)
    mse_prbf = mse(preds["prbf"], yte)
    mse_rbf = mse(preds["rbf"], yte)

    log_lines = [
        "Gray-Scott KRR benchmark",
        f"output_dir={out_dir}",
        f"grid={args.grid} steps={args.steps} burn_in={args.burn_in} frames={args.frames}",
        f"per_frame={args.per_frame} seed={args.seed} lam={args.lam}",
        f"du={args.du} dv={args.dv} f={args.f} k={args.k} dt={args.dt}",
        f"temperature={args.temperature} spatial_ell={args.spatial_ell} state_ell={args.state_ell}",
        f"train_samples={len(Xtr)} test_samples={len(Xte)}",
        f"MSE dashifine={mse_dash:.6f}",
        f"MSE periodic_rbf={mse_prbf:.6f}",
        f"MSE rbf={mse_rbf:.6f}",
        f"Top eigs dashifine={np.round(eig_dash[:10], 4)}",
        f"Top eigs periodic_rbf={np.round(eig_prbf[:10], 4)}",
        f"Top eigs rbf={np.round(eig_rbf[:10], 4)}",
    ]
    summary_path = out_dir / "run_summary.txt"
    rollout_csv = out_dir / "rollout_metrics.csv"
    rollout_plot = out_dir / "rollout_mse.png"
    rollout_gif = out_dir / "rollout.gif"
    log_lines.extend(
        [
            f"rollout_steps={args.rollout_steps}",
            f"rollout_snapshot_steps={args.rollout_snapshot_steps}",
            f"rollout_csv={rollout_csv}",
            f"rollout_plot={rollout_plot}",
            f"rollout_gif_steps={args.rollout_gif_steps}",
            f"rollout_gif_stride={args.rollout_gif_stride}",
            f"rollout_gif_fps={args.rollout_gif_fps}",
            f"rollout_gif={rollout_gif}",
        ]
    )
    summary_path.write_text("\n".join(log_lines))

    for line in log_lines:
        print(line)

    # Field snapshot comparison on the latest sampled frame.
    frame_idx = int(sample_frames[0])
    u_t, v_t = frames[frame_idx]
    u_next, v_next = frames[frame_idx + 1]
    xs = np.linspace(-np.pi, np.pi, args.grid, endpoint=False, dtype=np.float32)
    ys = np.linspace(-np.pi, np.pi, args.grid, endpoint=False, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([XX, YY], axis=-1).reshape(-1, 2)
    uv = np.stack([u_t, v_t], axis=-1).reshape(-1, 2)
    X_full = np.concatenate([coords, uv], axis=1)

    full_preds = {
        "dashifine": krr_fit_predict(Xtr, ytr, X_full, kernel_dash, args.lam),
        "prbf": krr_fit_predict(Xtr, ytr, X_full, kernel_prbf, args.lam),
    }
    save_field_comparison(
        args.grid,
        coords,
        u_next,
        v_next,
        full_preds,
        out_dir / "field_comparison.png",
    )

    snapshot_steps = []
    if args.rollout_snapshot_steps.strip():
        snapshot_steps = [
            int(s) for s in args.rollout_snapshot_steps.split(",") if s.strip()
        ]
    snapshot_steps = sorted({s for s in snapshot_steps if s > 0})

    gif_steps = max(0, int(args.rollout_gif_steps))
    gif_stride = max(1, int(args.rollout_gif_stride))
    gif_frames = []
    gif_dir = out_dir / "rollout_gif_frames"
    if gif_steps > 0:
        gif_dir.mkdir(parents=True, exist_ok=True)

    max_rollout = min(args.rollout_steps, args.steps - frame_idx - 1)
    if max_rollout < args.rollout_steps:
        print(
            f"Rollout truncated to {max_rollout} steps (not enough frames for {args.rollout_steps})."
        )

    u_true_seq = [u_t.copy()]
    v_true_seq = [v_t.copy()]
    for step in range(1, max_rollout + 1):
        u_true_seq.append(frames[frame_idx + step][0])
        v_true_seq.append(frames[frame_idx + step][1])

    coords_full = coords
    radii = np.sqrt(np.sum(coords_full**2, axis=1))
    radial_bins = max(4, args.grid // 2)
    radial_edges = np.linspace(0.0, radii.max(), radial_bins + 1)
    state_dash = uv.copy()
    state_prbf = uv.copy()
    state_rbf = uv.copy()

    rollout_rows = []
    for step in range(1, max_rollout + 1):
        X_dash = np.concatenate([coords_full, state_dash], axis=1)
        X_prbf = np.concatenate([coords_full, state_prbf], axis=1)
        X_rbf = np.concatenate([coords_full, state_rbf], axis=1)

        pred_dash = krr_fit_predict(Xtr, ytr, X_dash, kernel_dash, args.lam)
        pred_prbf = krr_fit_predict(Xtr, ytr, X_prbf, kernel_prbf, args.lam)
        pred_rbf = krr_fit_predict(Xtr, ytr, X_rbf, kernel_rbf, args.lam)

        u_true_step = u_true_seq[step]
        v_true_step = v_true_seq[step]
        uv_true = np.stack([u_true_step, v_true_step], axis=-1).reshape(-1, 2)

        mse_dash_roll = mse(pred_dash, uv_true)
        mse_prbf_roll = mse(pred_prbf, uv_true)
        mse_rbf_roll = mse(pred_rbf, uv_true)
        mse_u_dash = mse(pred_dash[:, 0], uv_true[:, 0])
        mse_v_dash = mse(pred_dash[:, 1], uv_true[:, 1])
        mse_u_prbf = mse(pred_prbf[:, 0], uv_true[:, 0])
        mse_v_prbf = mse(pred_prbf[:, 1], uv_true[:, 1])
        mse_u_rbf = mse(pred_rbf[:, 0], uv_true[:, 0])
        mse_v_rbf = mse(pred_rbf[:, 1], uv_true[:, 1])
        radial_true = radial_profile(uv_true[:, 0], radii, radial_edges)
        radial_dash = radial_profile(pred_dash[:, 0], radii, radial_edges)
        radial_prbf = radial_profile(pred_prbf[:, 0], radii, radial_edges)
        radial_rbf = radial_profile(pred_rbf[:, 0], radii, radial_edges)
        mse_radial_dash = mse(radial_dash, radial_true)
        mse_radial_prbf = mse(radial_prbf, radial_true)
        mse_radial_rbf = mse(radial_rbf, radial_true)

        mean_u_true = float(np.mean(u_true_step))
        mean_v_true = float(np.mean(v_true_step))
        mean_u_dash = float(np.mean(pred_dash[:, 0]))
        mean_v_dash = float(np.mean(pred_dash[:, 1]))
        mean_u_prbf = float(np.mean(pred_prbf[:, 0]))
        mean_v_prbf = float(np.mean(pred_prbf[:, 1]))
        mean_u_rbf = float(np.mean(pred_rbf[:, 0]))
        mean_v_rbf = float(np.mean(pred_rbf[:, 1]))

        mass_true = float(np.mean(u_true_step + v_true_step))
        mass_dash = float(np.mean(pred_dash[:, 0] + pred_dash[:, 1]))
        mass_prbf = float(np.mean(pred_prbf[:, 0] + pred_prbf[:, 1]))
        mass_rbf = float(np.mean(pred_rbf[:, 0] + pred_rbf[:, 1]))

        rollout_rows.append(
            [
                step,
                mse_dash_roll,
                mse_prbf_roll,
                mse_rbf_roll,
                mse_u_dash,
                mse_v_dash,
                mse_u_prbf,
                mse_v_prbf,
                mse_u_rbf,
                mse_v_rbf,
                mse_radial_dash,
                mse_radial_prbf,
                mse_radial_rbf,
                mean_u_true,
                mean_v_true,
                mean_u_dash,
                mean_v_dash,
                mean_u_prbf,
                mean_v_prbf,
                mean_u_rbf,
                mean_v_rbf,
                mass_true,
                mass_dash,
                mass_prbf,
                mass_rbf,
            ]
        )

        if step in snapshot_steps:
            save_rollout_snapshot(
                args.grid,
                u_true_step,
                v_true_step,
                {"dashifine": pred_dash, "prbf": pred_prbf, "rbf": pred_rbf},
                out_dir / f"rollout_step_{step:03d}.png",
            )
        if gif_steps > 0 and step <= gif_steps and step % gif_stride == 0:
            frame_path = gif_dir / f"rollout_step_{step:03d}.png"
            save_rollout_snapshot(
                args.grid,
                u_true_step,
                v_true_step,
                {"dashifine": pred_dash, "prbf": pred_prbf, "rbf": pred_rbf},
                frame_path,
            )
            gif_frames.append(frame_path)

        state_dash = pred_dash
        state_prbf = pred_prbf
        state_rbf = pred_rbf

    header = [
        "step",
        "mse_dashifine",
        "mse_prbf",
        "mse_rbf",
        "mse_u_dashifine",
        "mse_v_dashifine",
        "mse_u_prbf",
        "mse_v_prbf",
        "mse_u_rbf",
        "mse_v_rbf",
        "mse_u_radial_dashifine",
        "mse_u_radial_prbf",
        "mse_u_radial_rbf",
        "mean_u_true",
        "mean_v_true",
        "mean_u_dashifine",
        "mean_v_dashifine",
        "mean_u_prbf",
        "mean_v_prbf",
        "mean_u_rbf",
        "mean_v_rbf",
        "mass_true",
        "mass_dashifine",
        "mass_prbf",
        "mass_rbf",
    ]
    with rollout_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rollout_rows:
            f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + "\n")

    if rollout_rows:
        steps = np.array([row[0] for row in rollout_rows])
        mse_dash_vals = np.array([row[1] for row in rollout_rows])
        mse_prbf_vals = np.array([row[2] for row in rollout_rows])
        mse_rbf_vals = np.array([row[3] for row in rollout_rows])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, mse_dash_vals, label="Dashifine", lw=1.8)
        ax.plot(steps, mse_prbf_vals, label="pRBF", lw=1.8)
        ax.plot(steps, mse_rbf_vals, label="RBF", lw=1.8)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("MSE")
        ax.set_title("Gray-Scott rollout error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(rollout_plot, dpi=150)
        plt.close(fig)

    if gif_steps > 0:
        if imageio is None:
            print("imageio not available; skipping GIF export.")
        elif not gif_frames:
            print("No GIF frames captured; skipping GIF export.")
        else:
            duration = 1.0 / max(1, args.rollout_gif_fps)
            imageio.mimsave(rollout_gif, [imageio.imread(p) for p in gif_frames], duration=duration)
            if not args.keep_gif_frames:
                for p in gif_frames:
                    p.unlink(missing_ok=True)
                try:
                    gif_dir.rmdir()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
