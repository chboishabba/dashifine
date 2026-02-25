# 26_grok_sweep_adaptive_spv.py
from __future__ import annotations
import argparse, csv, math, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
device = "cpu"

# -----------------------------
# Model
# -----------------------------
class ModMLP(nn.Module):
    def __init__(self, p: int, d: int = 128, h: int = 512):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.fc1 = nn.Linear(2 * d, h)
        self.fc2 = nn.Linear(h, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ea = self.emb(x[:, 0])
        eb = self.emb(x[:, 1])
        z = torch.cat([ea, eb], dim=-1)
        z = F.relu(self.fc1(z))
        return self.fc2(z)

@torch.no_grad()
def eval_acc(model, x, y):
    model.eval()
    return (model(x).argmax(-1) == y).float().mean().item()

def first_epoch_at_or_none(epochs_logged, values, thr):
    for e, v in zip(epochs_logged, values):
        if v >= thr:
            return e
    return None

# -----------------------------
# Single run
# -----------------------------
def run_one(seed, weight_decay,
            p=97,
            train_frac=0.3,
            epochs=40000,
            log_every=50,
            lr=1e-3,
            d=128,
            h=512):

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

    epochs_logged, train_log, test_log = [], [], []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        opt.step()

        if ep == 1 or ep % log_every == 0:
            tr = eval_acc(model, Xtr, ytr)
            te = eval_acc(model, Xte, yte)
            epochs_logged.append(ep)
            train_log.append(tr)
            test_log.append(te)

    t_fit = first_epoch_at_or_none(epochs_logged, train_log, 0.99)
    t95 = first_epoch_at_or_none(epochs_logged, test_log, 0.95)

    return {
        "seed": seed,
        "weight_decay": weight_decay,
        "t_fit": t_fit,
        "t95": t95,
        "final_train_acc": train_log[-1],
        "final_test_acc": test_log[-1],
    }

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=40000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(3)))
    p.add_argument("--coarse-wds", type=float, nargs="+",
                   default=[1.0, 0.0, 0.02, 0.05, 0.08, 0.12, 0.2])
    p.add_argument("--fine-radius", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--p", type=int, default=97)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--out", type=str, default="grok_sweep_gpu.csv")
    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    global device
    args = parse_args()

    if args.torch_device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.torch_device

    print(f"[torch] using device: {device}")

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

    results = []

    print("\n--- Phase 1: Coarse Scan ---\n")

    for wd in args.coarse_wds:
        for seed in args.seeds:
            r = run_one(seed=seed,
                        weight_decay=wd,
                        p=args.p,
                        train_frac=args.train_frac,
                        epochs=args.epochs,
                        log_every=args.log_every,
                        lr=args.lr)
            results.append(r)
            print(f"wd={wd:<5} seed={seed:<2} "
                  f"t95={r['t95']} "
                  f"train={r['final_train_acc']:.3f} "
                  f"test={r['final_test_acc']:.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
