import math, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
device = "cpu"

# -------------------------
# Model
# -------------------------
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
def eval_metrics(model, x, y):
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc

def l2_norm(model):
    return math.sqrt(sum((p.detach()**2).sum().item() for p in model.parameters()))

def spectral_norm_last_layer(model):
    W = model.fc2.weight.detach()
    s = torch.linalg.svdvals(W)
    return s.max().item()

def first_epoch_at_or_none(epochs_logged, values, thr):
    for e, v in zip(epochs_logged, values):
        if v >= thr:
            return e
    return None

def run_one(seed, p=97, train_frac=0.3, epochs=20000, log_every=20, lr=1e-3, weight_decay=0.08, d=128, h=512):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # data
    pairs = [(a,b) for a in range(p) for b in range(p)]
    N = len(pairs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(train_frac * N)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    ab = np.array(pairs, dtype=np.int64)
    y_all = (ab[:,0] * ab[:,1]) % p

    Xtr = torch.tensor(ab[train_idx], device=device)
    ytr = torch.tensor(y_all[train_idx], device=device)
    Xte = torch.tensor(ab[test_idx], device=device)
    yte = torch.tensor(y_all[test_idx], device=device)

    model = ModMLP(p=p, d=d, h=h).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"epoch": [], "train_acc": [], "test_acc": [], "l2": [], "spec": []}

    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        opt.step()

        if ep == 1 or ep % log_every == 0:
            _, tr_acc = eval_metrics(model, Xtr, ytr)
            _, te_acc = eval_metrics(model, Xte, yte)
            hist["epoch"].append(ep)
            hist["train_acc"].append(tr_acc)
            hist["test_acc"].append(te_acc)
            hist["l2"].append(l2_norm(model))
            hist["spec"].append(spectral_norm_last_layer(model))

    # thresholds
    a0 = 1.0 / p
    t_fit = first_epoch_at_or_none(hist["epoch"], hist["train_acc"], 0.99)
    t10  = first_epoch_at_or_none(hist["epoch"], hist["test_acc"], max(0.10, 5*a0))
    t50  = first_epoch_at_or_none(hist["epoch"], hist["test_acc"], 0.50)
    t95  = first_epoch_at_or_none(hist["epoch"], hist["test_acc"], 0.95)
    width = None if (t10 is None or t95 is None) else (t95 - t10)

    return {
        "seed": seed, "p": p, "train_frac": train_frac, "epochs": epochs, "log_every": log_every,
        "lr": lr, "weight_decay": weight_decay, "d": d, "h": h,
        "t_fit": t_fit, "t10": t10, "t50": t50, "t95": t95, "width": width,
        "final_train_acc": hist["train_acc"][-1],
        "final_test_acc":  hist["test_acc"][-1],
        "final_l2":         hist["l2"][-1],
        "final_spec":       hist["spec"][-1],
    }

def main():
    p = 97
    train_frac = 0.3
    epochs = 20000
    log_every = 20
    lr = 1e-3
    d, h = 128, 512

    wds = [0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0]
    seeds = list(range(10))  # increase to 20 if you want

    out_csv = "grok_sweep_results.csv"
    rows = []

    for wd in wds:
        for seed in seeds:
            r = run_one(seed=seed, p=p, train_frac=train_frac, epochs=epochs, log_every=log_every,
                        lr=lr, weight_decay=wd, d=d, h=h)
            rows.append(r)
            print(f"wd={wd:>4} seed={seed:>2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    # write CSV
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote: {out_csv}")

if __name__ == "__main__":
    main()
