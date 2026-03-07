"""
Critical-window grokking refinement scan for modular multiplication.

This runner is a lower-weight-decay follow-up to `26_grok_critical_scan.py`.
It writes:
  - `2_grok_critical_scan_refine.csv` with one row per completed run
  - `2_grok_critical_scan_refine_trajectories.csv` with per-epoch train/test metrics

Runs are checkpointed after each completed `(p, weight_decay, seed)` tuple so
the scan can be resumed safely. A run stops early only after the test accuracy
has remained above the grok threshold for several logged checkpoints.
"""

import math
import random
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
device = "cpu"


class ModMLP(nn.Module):
    def __init__(self, p, d=128, h=512):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.fc1 = nn.Linear(2 * d, h)
        self.fc2 = nn.Linear(h, p)

    def forward(self, x):
        ea = self.emb(x[:, 0])
        eb = self.emb(x[:, 1])
        z = torch.cat([ea, eb], dim=-1)
        z = F.relu(self.fc1(z))
        return self.fc2(z)


@torch.no_grad()
def eval_acc(model, x, y):
    model.eval()
    return (model(x).argmax(-1) == y).float().mean().item()


@torch.no_grad()
def eval_loss(model, x, y):
    model.eval()
    return float(F.cross_entropy(model(x), y).item())


def first_epoch_at_or_none(epochs_logged, values, thr):
    for e, v in zip(epochs_logged, values):
        if v >= thr:
            return e
    return None


def run_one(seed, weight_decay, p, train_frac, epochs, log_every, lr, d, h):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pairs = [(a, b) for a in range(p) for b in range(p)]
    N = len(pairs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(train_frac * N)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    ab = np.array(pairs, dtype=np.int64)
    y_all = (ab[:, 0] * ab[:, 1]) % p
    Xtr = torch.tensor(ab[train_idx], device=device)
    ytr = torch.tensor(y_all[train_idx], device=device)
    Xte = torch.tensor(ab[test_idx], device=device)
    yte = torch.tensor(y_all[test_idx], device=device)

    model = ModMLP(p=p, d=d, h=h).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs_logged = []
    train_acc_log = []
    test_acc_log = []
    train_loss_log = []
    test_loss_log = []
    grok_patience_logs = 5
    grok_thr = 0.95

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        opt.step()

        if ep == 1 or ep % log_every == 0:
            tr_loss = eval_loss(model, Xtr, ytr)
            te_loss = eval_loss(model, Xte, yte)
            tr_acc = eval_acc(model, Xtr, ytr)
            te_acc = eval_acc(model, Xte, yte)
            epochs_logged.append(ep)
            train_loss_log.append(tr_loss)
            test_loss_log.append(te_loss)
            train_acc_log.append(tr_acc)
            test_acc_log.append(te_acc)

            if len(test_acc_log) >= grok_patience_logs:
                tail = test_acc_log[-grok_patience_logs:]
                if min(tail) >= grok_thr:
                    break

    t_fit = first_epoch_at_or_none(epochs_logged, train_acc_log, 0.99)
    t95 = first_epoch_at_or_none(epochs_logged, test_acc_log, 0.95)

    summary = {
        "p": p,
        "seed": seed,
        "weight_decay": float(weight_decay),
        "epochs": epochs,
        "train_frac": train_frac,
        "lr": lr,
        "d": d,
        "h": h,
        "t_fit": t_fit,
        "t95": t95,
        "final_train_loss": train_loss_log[-1],
        "final_test_loss": test_loss_log[-1],
        "final_train_acc": train_acc_log[-1],
        "final_test_acc": test_acc_log[-1],
    }
    trajectory = []
    for ep, tr_loss, te_loss, tr_acc, te_acc in zip(
        epochs_logged, train_loss_log, test_loss_log, train_acc_log, test_acc_log
    ):
        trajectory.append({
            "p": p,
            "seed": seed,
            "weight_decay": float(weight_decay),
            "epoch": ep,
            "train_loss": tr_loss,
            "test_loss": te_loss,
            "train_acc": tr_acc,
            "test_acc": te_acc,
        })
    return summary, trajectory


def main():
    train_frac = 0.3
    lr = 1e-3
    d, h = 128, 512
    log_every = 20

    p_main = 97
    epochs_main = 50000
    seeds_main = [0]
    wds_main = [0.20, 0.22, 0.24]

    primes_extra = []
    epochs_extra = 20000
    seeds_extra = list(range(5))
    wds_extra = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    out = Path("2_grok_critical_scan_refine.csv")
    traj_out = Path("2_grok_critical_scan_refine_trajectories.csv")

    completed = set()
    if out.exists():
        with out.open("r", newline="") as f:
            for row in csv.DictReader(f):
                completed.add((int(row["p"]), float(row["weight_decay"]), int(row["seed"])))

    summary_fields = [
        "p",
        "seed",
        "weight_decay",
        "epochs",
        "train_frac",
        "lr",
        "d",
        "h",
        "t_fit",
        "t95",
        "final_train_loss",
        "final_test_loss",
        "final_train_acc",
        "final_test_acc",
    ]
    traj_fields = [
        "p",
        "seed",
        "weight_decay",
        "epoch",
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
    ]

    if not out.exists():
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
    if not traj_out.exists():
        with traj_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=traj_fields)
            w.writeheader()

    def persist(summary, traj):
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writerow(summary)
        with traj_out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=traj_fields)
            w.writerows(traj)

    print("\n=== Refine scan p=97 ===\n")
    for wd in wds_main:
        for seed in seeds_main:
            key = (p_main, float(wd), seed)
            if key in completed:
                print(f"p={p_main} wd={wd:<4} seed={seed:<2} already done, skipping")
                continue
            r, traj = run_one(seed, wd, p_main, train_frac, epochs_main, log_every, lr, d, h)
            persist(r, traj)
            print(f"p={p_main} wd={wd:<4} seed={seed:<2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    for p in primes_extra:
        print(f"\n=== Cross-prime scan p={p} ===\n")
        for wd in wds_extra:
            for seed in seeds_extra:
                key = (p, float(wd), seed)
                if key in completed:
                    print(f"p={p} wd={wd:<4} seed={seed:<2} already done, skipping")
                    continue
                r, traj = run_one(seed, wd, p, train_frac, epochs_extra, log_every, lr, d, h)
                persist(r, traj)
                print(f"p={p} wd={wd:<4} seed={seed:<2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    print(f"\nSaved: {out}")
    print(f"Saved: {traj_out}")


if __name__ == "__main__":
    main()
