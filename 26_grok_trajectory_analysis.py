#!/usr/bin/env python3
"""
Analyze checkpointed grokking trajectories.

Inputs:
  - grok_critical_scan.csv
  - grok_critical_scan_trajectories.csv

Outputs (default: grok_analysis/):
  - grok_milestones.csv
  - grok_onset_fit_screen.csv
  - grok_test_acc_raw.png
  - grok_test_loss_raw.png
  - grok_test_acc_norm_t50.png
  - grok_test_acc_norm_t95.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


TrajectoryKey = Tuple[int, float, int]


def load_summary(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "p": int(row["p"]),
                "seed": int(row["seed"]),
                "weight_decay": float(row["weight_decay"]),
                "epochs": float(row["epochs"]),
                "train_frac": float(row["train_frac"]),
                "lr": float(row["lr"]),
                "d": float(row["d"]),
                "h": float(row["h"]),
                "t_fit": float(row["t_fit"]) if row["t_fit"] else math.nan,
                "t95": float(row["t95"]) if row["t95"] else math.nan,
                "final_train_loss": float(row["final_train_loss"]),
                "final_test_loss": float(row["final_test_loss"]),
                "final_train_acc": float(row["final_train_acc"]),
                "final_test_acc": float(row["final_test_acc"]),
            })
    return rows


def load_trajectories(path: Path) -> Dict[TrajectoryKey, List[Dict[str, float]]]:
    out: Dict[TrajectoryKey, List[Dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
            out[key].append({
                "epoch": float(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "test_loss": float(row["test_loss"]),
                "train_acc": float(row["train_acc"]),
                "test_acc": float(row["test_acc"]),
            })
    for rows in out.values():
        rows.sort(key=lambda r: r["epoch"])
    return out


def first_ge(rows: Iterable[Dict[str, float]], field: str, thr: float) -> float:
    for row in rows:
        if row[field] >= thr:
            return row["epoch"]
    return math.nan


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fit_linear(
    xs: List[float], ys: List[float]
) -> Tuple[float, float, float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx == 0.0:
        return (math.nan, math.nan, math.nan)
    slope = float(np.sum((x - xbar) * (y - ybar)) / sxx)
    intercept = ybar - slope * xbar
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r2


def interpolate(xs: List[float], ys: List[float], grid: List[float]) -> List[float]:
    out: List[float] = []
    for g in grid:
        if g <= xs[0]:
            out.append(ys[0])
            continue
        if g >= xs[-1]:
            out.append(ys[-1])
            continue
        j = 1
        while xs[j] < g:
            j += 1
        x0, x1 = xs[j - 1], xs[j]
        y0, y1 = ys[j - 1], ys[j]
        t = (g - x0) / (x1 - x0)
        out.append(y0 + t * (y1 - y0))
    return out


def normalized_alignment_mse(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    milestone_field: str,
) -> float:
    grid = [i / 20.0 for i in range(21)]
    curves: List[List[float]] = []
    for row in milestones:
        T = float(row[milestone_field])
        if not math.isfinite(T) or T <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        xs = [pt["epoch"] / T for pt in traj]
        ys = [pt["test_acc"] for pt in traj]
        curves.append(interpolate(xs, ys, grid))
    if not curves:
        return math.nan
    mean = [sum(curve[i] for curve in curves) / len(curves) for i in range(len(grid))]
    return sum((curve[i] - mean[i]) ** 2 for curve in curves for i in range(len(grid))) / (
        len(curves) * len(grid)
    )


def plot_raw(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    y_field: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in sorted(trajectories):
        p, wd, seed = key
        rows = trajectories[key]
        ax.plot(
            [r["epoch"] for r in rows],
            [r[y_field] for r in rows],
            label=f"p={p}, wd={wd:.2f}, seed={seed}",
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_field.replace("_", " "))
    ax.set_title(f"Raw {y_field.replace('_', ' ')} trajectories")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_normalized(
    trajectories: Dict[TrajectoryKey, List[Dict[str, float]]],
    milestones: List[Dict[str, object]],
    milestone_field: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in sorted(milestones, key=lambda r: float(r["weight_decay"])):
        T = float(row[milestone_field])
        if not math.isfinite(T) or T <= 0:
            continue
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        ax.plot(
            [pt["epoch"] / T for pt in traj],
            [pt["test_acc"] for pt in traj],
            label=f"wd={float(row['weight_decay']):.2f}",
            linewidth=2,
        )
    ax.set_xlabel(f"Epoch / {milestone_field}")
    ax.set_ylabel("test acc")
    ax.set_title(f"Normalized test acc by {milestone_field}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="grok_critical_scan.csv")
    ap.add_argument("--trajectories", default="grok_critical_scan_trajectories.csv")
    ap.add_argument("--outdir", default="grok_analysis")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    traj_path = Path(args.trajectories)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_summary(summary_path)
    trajectories = load_trajectories(traj_path)
    if not summary_rows or not trajectories:
        raise SystemExit("missing grok summary or trajectory rows")

    milestone_rows: List[Dict[str, object]] = []
    for row in sorted(summary_rows, key=lambda r: (r["p"], r["weight_decay"], r["seed"])):
        key = (int(row["p"]), float(row["weight_decay"]), int(row["seed"]))
        traj = trajectories[key]
        milestone_rows.append({
            "p": int(row["p"]),
            "seed": int(row["seed"]),
            "weight_decay": float(row["weight_decay"]),
            "t_fit": int(row["t_fit"]) if math.isfinite(row["t_fit"]) else "",
            "t10": int(first_ge(traj, "test_acc", 0.10)),
            "t20": int(first_ge(traj, "test_acc", 0.20)),
            "t50": int(first_ge(traj, "test_acc", 0.50)),
            "t80": int(first_ge(traj, "test_acc", 0.80)),
            "t90": int(first_ge(traj, "test_acc", 0.90)),
            "t95": int(first_ge(traj, "test_acc", 0.95)),
            "final_test_acc": float(row["final_test_acc"]),
            "final_test_loss": float(row["final_test_loss"]),
        })

    write_csv(
        outdir / "grok_milestones.csv",
        [
            "p",
            "seed",
            "weight_decay",
            "t_fit",
            "t10",
            "t20",
            "t50",
            "t80",
            "t90",
            "t95",
            "final_test_acc",
            "final_test_loss",
        ],
        milestone_rows,
    )

    fit_specs: List[Tuple[str, Callable[[float], float], Callable[[float], float]]] = [
        ("t95 ~ wd", lambda wd: wd, lambda t: t),
        ("t95 ~ 1/wd", lambda wd: 1.0 / wd, lambda t: t),
        ("log t95 ~ wd", lambda wd: wd, lambda t: math.log(t)),
        ("log t95 ~ 1/wd", lambda wd: 1.0 / wd, lambda t: math.log(t)),
    ]
    fit_rows: List[Dict[str, object]] = []
    xs_wd = [float(row["weight_decay"]) for row in milestone_rows]
    ys_t95 = [float(row["t95"]) for row in milestone_rows]
    for label, xf, yf in fit_specs:
        xs = [xf(wd) for wd in xs_wd]
        ys = [yf(t95) for t95 in ys_t95]
        intercept, slope, r2 = fit_linear(xs, ys)
        fit_rows.append({
            "model": label,
            "intercept": intercept,
            "slope": slope,
            "r2": r2,
        })
    fit_rows.append({
        "model": "alignment_mse_norm_t50",
        "intercept": "",
        "slope": "",
        "r2": normalized_alignment_mse(trajectories, milestone_rows, "t50"),
    })
    fit_rows.append({
        "model": "alignment_mse_norm_t95",
        "intercept": "",
        "slope": "",
        "r2": normalized_alignment_mse(trajectories, milestone_rows, "t95"),
    })
    write_csv(
        outdir / "grok_onset_fit_screen.csv",
        ["model", "intercept", "slope", "r2"],
        fit_rows,
    )

    plot_raw(trajectories, "test_acc", outdir / "grok_test_acc_raw.png")
    plot_raw(trajectories, "test_loss", outdir / "grok_test_loss_raw.png")
    plot_normalized(trajectories, milestone_rows, "t50", outdir / "grok_test_acc_norm_t50.png")
    plot_normalized(trajectories, milestone_rows, "t95", outdir / "grok_test_acc_norm_t95.png")

    print(f"[ok] wrote {outdir / 'grok_milestones.csv'}")
    print(f"[ok] wrote {outdir / 'grok_onset_fit_screen.csv'}")
    print(f"[ok] wrote {outdir / 'grok_test_acc_raw.png'}")
    print(f"[ok] wrote {outdir / 'grok_test_loss_raw.png'}")
    print(f"[ok] wrote {outdir / 'grok_test_acc_norm_t50.png'}")
    print(f"[ok] wrote {outdir / 'grok_test_acc_norm_t95.png'}")


if __name__ == "__main__":
    main()
