"""Reconstruct archived conversational path plot (Self, Norm, Mirror).

Generates three figures:
1) 3D trajectory through the 27-state lattice.
2) Projection onto the mirror-averaged resonance surface z = self*norm.
3) Optional GELU-odd neutral surface variant.

Outputs PNGs beside this script.
"""

import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def enc(sym: str) -> int:
    return {"+": 1, "0": 0, "-": -1}[sym]


def build_path():
    phase1_sym = [
        ("A", ("+", "+", "+")),
        ("B", ("+", "-", "0")),
        ("C", ("+", "-", "-")),
        ("D", ("-", "-", "+")),
        ("E", ("-", "0", "+")),
    ]
    phase2_sym = [
        ("E1", ("-", "-", "+")),
        ("F", ("0", "-", "+")),
        ("G", ("+", "0", "+")),
        ("H", ("+", "+", "0")),
        ("I", ("+", "+", "+")),
    ]
    path_sym = phase1_sym + phase2_sym
    return [(lab, enc(a), enc(b), enc(c)) for lab, (a, b, c) in path_sym]


def interpolate_path(points, n_per_seg=80):
    traj = []
    labels = []
    for i in range(len(points) - 1):
        lab_i, x1, y1, z1 = points[i]
        lab_j, x2, y2, z2 = points[i + 1]
        ts = np.linspace(0, 1, n_per_seg, endpoint=False)
        seg = np.column_stack(
            [x1 + ts * (x2 - x1), y1 + ts * (y2 - y1), z1 + ts * (z2 - z1)]
        )
        traj.append(seg)
        labels.extend([lab_i] * len(ts))
    traj.append(np.array([[points[-1][1], points[-1][2], points[-1][3]]]))
    labels.append(points[-1][0])
    return np.vstack(traj), labels


def plot_trajectory(path, traj):
    levels = [-1, 0, 1]
    grid = np.array([(x, y, z) for x in levels for y in levels for z in levels])

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], alpha=0.6)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2.0)
    for lab, x, y, z in path:
        ax.scatter([x], [y], [z], s=60)
        ax.text(x, y, z, f" {lab}", zdir=None)
    ax.set_xlabel("Self (3)  -1 … 0 … +1")
    ax.set_ylabel("Norm (4)  -1 … 0 … +1")
    ax.set_zlabel("Mirror (5) -1 … 0 … +1")
    ax.set_title("Full Conversational Path in (Self, Norm, Mirror) Space")
    fig.tight_layout()
    fig.savefig("full_path_3d.png", dpi=200)
    plt.close(fig)


def plot_resonance(path, traj):
    s = np.linspace(-1, 1, 200)
    S3, S4 = np.meshgrid(s, s)
    Z = S3 * S4

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S3, S4, Z, rstride=4, cstride=4, linewidth=0, antialiased=True)
    Zpath = traj[:, 0] * traj[:, 1]
    ax.plot(traj[:, 0], traj[:, 1], Zpath, linewidth=2.0, marker="o", markersize=2)
    for lab, x, y, _ in path:
        ax.scatter([x], [y], [x * y], s=50)
        ax.text(x, y, x * y, f" {lab}", zdir=None)
    ax.set_xlabel("Self (3)  -1 … 0 … +1")
    ax.set_ylabel("Norm (4)  -1 … 0 … +1")
    ax.set_zlabel("Mean resonance  z = self·norm")
    ax.set_title("Full Path on Mirror-Averaged Resonance Surface")
    fig.tight_layout()
    fig.savefig("full_path_resonance.png", dpi=200)
    plt.close(fig)


def plot_gelu_surface(path, traj):
    s = np.linspace(-1, 1, 200)
    S3, S4 = np.meshgrid(s, s)

    def gelu(x: float) -> float:
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

    vgelu = np.vectorize(gelu)

    k = 2.0
    _A = vgelu(k * (S3 + S4)) - vgelu(-k * (S3 + S4))  # kept neutral (m=0)
    Zg = S3 * S4

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S3, S4, Zg, rstride=4, cstride=4, linewidth=0, antialiased=True)
    Zg_path = traj[:, 0] * traj[:, 1]
    ax.plot(traj[:, 0], traj[:, 1], Zg_path, linewidth=2.0, marker="o", markersize=2)
    for lab, x, y, _ in path:
        ax.scatter([x], [y], [x * y], s=50)
        ax.text(x, y, x * y, f" {lab}", zdir=None)
    ax.set_xlabel("Self (3)  -1 … 0 … +1")
    ax.set_ylabel("Norm (4)  -1 … 0 … +1")
    ax.set_zlabel("GELU-odd surface  z = self·norm")
    ax.set_title("Full Path on GELU-odd (neutral) Surface")
    fig.tight_layout()
    fig.savefig("full_path_gelu.png", dpi=200)
    plt.close(fig)


def emit_table(path):
    def resonance(x, y, z):
        return x * y + y * z + z * x

    rows = [
        {
            "label": lab,
            "self_3": x,
            "norm_4": y,
            "mirror_5": z,
            "R = s3*s4 + s4*s5 + s5*s3": resonance(x, y, z),
            "z = self*norm": x * y,
        }
        for lab, x, y, z in path
    ]
    df = pd.DataFrame(rows)
    df.to_csv("full_path_metrics.csv", index=False)
    return df


def main():
    path = build_path()
    traj, _ = interpolate_path(path, n_per_seg=80)
    plot_trajectory(path, traj)
    plot_resonance(path, traj)
    plot_gelu_surface(path, traj)
    emit_table(path)


if __name__ == "__main__":
    main()

