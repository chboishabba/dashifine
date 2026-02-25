import math
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
device = "cpu"

# ======================
# Model
# ======================
class ModMLP(nn.Module):
    def __init__(self, p, d=128, h=512):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.fc1 = nn.Linear(2*d, h)
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

def first_epoch_at_or_none(epochs_logged, values, thr):
    for e, v in zip(epochs_logged, values):
        if v >= thr:
            return e
    return None

# ======================
# Single run
# ======================
def run_one(seed, weight_decay,
            p=97,
            train_frac=0.3,
            epochs=15000,
            log_every=20,
            lr=1e-3,
            d=128,
            h=512):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
    pairs = [(a, b) for a in range(p) for b in range(p)]
    N = len(pairs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_train = int(train_frac * N)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    ab = np.array(pairs, dtype=np.int64)
    y_all = (ab[:, 0] * ab[:, 1]) % p

    Xtr = torch.tensor(ab[train_idx], device=device)
    ytr = torch.tensor(y_all[train_idx], device=device)
    Xte = torch.tensor(ab[test_idx], device=device)
    yte = torch.tensor(y_all[test_idx], device=device)

    model = ModMLP(p=p, d=d, h=h).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epochs_logged = []
    train_acc_log = []
    test_acc_log  = []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        opt.step()

        if ep == 1 or ep % log_every == 0:
            tr_acc = eval_acc(model, Xtr, ytr)
            te_acc = eval_acc(model, Xte, yte)

            epochs_logged.append(ep)
            train_acc_log.append(tr_acc)
            test_acc_log.append(te_acc)

    t_fit = first_epoch_at_or_none(epochs_logged, train_acc_log, 0.99)
    t95   = first_epoch_at_or_none(epochs_logged, test_acc_log, 0.95)

    return {
        "seed": seed,
        "weight_decay": weight_decay,
        "t_fit": t_fit,
        "t95": t95,
        "final_test_acc": test_acc_log[-1]
    }

# ======================
# Adaptive Sweep
# ======================
def main():

    seeds = list(range(3))  # increase later

    # ----- Phase 1: coarse scan -----
    # coarse_wds = [0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.5, 0.8, 0,9 1.0]
    coarse_wds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    print("\n--- Phase 1: Coarse Scan ---\n")

    for wd in coarse_wds:
        for seed in seeds:
            r = run_one(seed=seed, weight_decay=wd)
            results.append(r)
            print(f"wd={wd:<5} seed={seed:<2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    # Identify first wd that groks
    grok_wds = sorted({r["weight_decay"] for r in results if r["t95"] is not None})

    if len(grok_wds) == 0:
        print("\nNo grokking detected in coarse scan.")
        return

    wd_min = min(grok_wds)
    print(f"\nSmallest wd that groks (coarse): {wd_min}")

    # ----- Phase 2: fine scan around transition -----
    lower = max(0.0, wd_min - 0.05)
    upper = wd_min + 0.05
    fine_wds = np.round(np.arange(lower, upper, 0.01), 3).tolist()

    print("\n--- Phase 2: Fine Scan ---\n")

    for wd in fine_wds:
        for seed in seeds:
            r = run_one(seed=seed, weight_decay=wd)
            results.append(r)
            print(f"wd={wd:<5} seed={seed:<2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    # Save CSV
    with open("grok_sweep_adaptive.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print("\nSaved: grok_sweep_adaptive.csv")

if __name__ == "__main__":
    main()
