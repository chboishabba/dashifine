import math
import random
import csv
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

def run_one(seed, weight_decay, p, train_frac, epochs, log_every, lr, d, h):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # data
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        "final_train_acc": train_acc_log[-1],
        "final_test_acc":  test_acc_log[-1],
    }

def main():
    # Shared hyperparams (keep fixed for comparability)
    train_frac = 0.3
    lr = 1e-3
    d, h = 128, 512
    log_every = 20

    # ---- Critical scan for p=97 ----
    p_main = 97
    epochs_main = 30000
    seeds_main = list(range(10))
    wds_main = [0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.60]

    # ---- Cross-prime sanity ----
    primes_extra = [47, 193]
    epochs_extra = 20000
    seeds_extra = list(range(5))
    wds_extra = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    rows = []

    print("\n=== Critical scan p=97 ===\n")
    for wd in wds_main:
        for seed in seeds_main:
            r = run_one(seed, wd, p_main, train_frac, epochs_main, log_every, lr, d, h)
            rows.append(r)
            print(f"p={p_main} wd={wd:<4} seed={seed:<2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    for p in primes_extra:
        print(f"\n=== Cross-prime scan p={p} ===\n")
        for wd in wds_extra:
            for seed in seeds_extra:
                r = run_one(seed, wd, p, train_frac, epochs_extra, log_every, lr, d, h)
                rows.append(r)
                print(f"p={p} wd={wd:<4} seed={seed:<2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    out = "grok_critical_scan.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
