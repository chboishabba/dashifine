import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------
# Reproducibility
# -----------------------
torch.set_num_threads(2)
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = "cpu"

# -----------------------
# Problem setup
# -----------------------
p = 97
train_frac = 0.3
epochs = 10000
log_every = 20

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

# -----------------------
# Model
# -----------------------
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

model = ModMLP(p=p).to(device)

# -----------------------
# Optimizer
# -----------------------
weight_decay = 0.0  # change to 0.08 or 1.0 to test regularization effects
opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=weight_decay
)

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def eval_metrics(x, y):
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc

def l2_norm():
    return math.sqrt(sum((p.detach() ** 2).sum().item() for p in model.parameters()))

def spectral_norm_last_layer():
    W = model.fc2.weight.detach()
    s = torch.linalg.svdvals(W)
    return s.max().item()

# -----------------------
# Logging
# -----------------------
hist = {
    "epoch": [],
    "train_acc": [],
    "test_acc": [],
    "train_loss": [],
    "test_loss": [],
    "l2": [],
    "spec": []
}

# -----------------------
# Training loop
# -----------------------
for ep in range(1, epochs + 1):
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(Xtr)
    loss = F.cross_entropy(logits, ytr)
    loss.backward()
    opt.step()

    if ep == 1 or ep % log_every == 0:
        tr_loss, tr_acc = eval_metrics(Xtr, ytr)
        te_loss, te_acc = eval_metrics(Xte, yte)

        hist["epoch"].append(ep)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        hist["train_loss"].append(tr_loss)
        hist["test_loss"].append(te_loss)
        hist["l2"].append(l2_norm())
        hist["spec"].append(spectral_norm_last_layer())

# -----------------------
# Plots
# -----------------------
plt.figure()
plt.plot(hist["epoch"], hist["train_acc"], label="train")
plt.plot(hist["epoch"], hist["test_acc"], label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Grokking Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist["epoch"], hist["l2"])
plt.xlabel("epoch")
plt.ylabel("L2 norm")
plt.title("Parameter Norm")
plt.show()

plt.figure()
plt.plot(hist["epoch"], hist["spec"])
plt.xlabel("epoch")
plt.ylabel("Spectral Norm (Last Layer)")
plt.title("Spectral Norm")
plt.show()

# -----------------------
# Summary
# -----------------------
summary = {
    "p": p,
    "N_total": N,
    "train_N": len(train_idx),
    "test_N": len(test_idx),
    "epochs": epochs,
    "weight_decay": weight_decay,
    "final_train_acc": hist["train_acc"][-1],
    "final_test_acc": hist["test_acc"][-1],
    "final_train_loss": hist["train_loss"][-1],
    "final_test_loss": hist["test_loss"][-1],
    "logged_points": len(hist["epoch"])
}

print(summary)
