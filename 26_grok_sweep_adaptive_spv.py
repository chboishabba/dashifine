"""
Adaptive grok sweep with an optional Vulkan/SPIR-V smoke test.

- Core sweep logic matches 26_grok_sweep_adaptive.py (Torch MLP on mod-mul).
- `--spv-demo` runs a tiny GEMV using the prebuilt SPIR-V shaders in
  ../dashiCORE/spv/. This checks that the dashiCORE Vulkan path is wired
  without changing training behaviour.

Run from the dashifine repo root (Torch on CPU by default):
  python 26_grok_sweep_adaptive_spv.py --spv-demo

If Vulkan/glslc/ICD are missing the SPV step is skipped gracefully.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
# Will be overridden in main() based on CLI flag/availability.
device = "cpu"

# Make dashiCORE modules importable when run from dashifine/
CORE_ROOT = Path(__file__).resolve().parents[1] / "dashiCORE"
if CORE_ROOT.exists() and str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

try:
    from gpu_common_methods import resolve_shader_candidates
    from gpu_vulkan_dispatcher import VulkanDispatchConfig
    from gpu_vulkan_gemv import VulkanGemvExecutor, has_vulkan

    HAS_DASHI_CORE = True
except Exception:
    HAS_DASHI_CORE = False


def run_spv_demo(N: int = 64, device_index: int = 0, verbose: bool = True) -> np.ndarray | None:
    """
    Dispatch a tiny GEMV via dashiCORE's prebuilt SPIR-V shaders.

    Uses gemv_tiled → gemv_or_rollout as shader candidates. Returns the output
    vector on success, or None on any import/device failure.
    """

    if not HAS_DASHI_CORE:
        if verbose:
            print("[spv] dashiCORE Vulkan path unavailable; skipping SPV demo.")
        return None

    if not has_vulkan():
        if verbose:
            print("[spv] Vulkan/python bindings not present; skipping SPV demo.")
        return None

    shader_path = resolve_shader_candidates(("gemv_tiled", "gemv_or_rollout"))
    dispatch_cfg = VulkanDispatchConfig(device_index=device_index)

    try:
        exec = VulkanGemvExecutor(
            N,
            shader_path=shader_path,
            dispatch_config=dispatch_cfg,
            timing_enabled=False,
        )
    except Exception as exc:  # pragma: no cover - hardware/env dependent
        if verbose:
            print(f"[spv] Failed to initialize Vulkan executor: {exc}")
        return None

    A = np.eye(N, dtype=np.float32)
    x = np.arange(N, dtype=np.float32)
    y = exec.gemv(A, x)
    timings = exec.get_last_timings()
    exec.close()

    if verbose:
        gpu_ms = timings.get("gpu_time_ms", 0.0)
        print(
            f"[spv] OK | shader={shader_path.name} | N={N} | y0={y[0]:.2f} | gpu_time_ms={gpu_ms:.3f}"
        )
    return y


# ======================
# Model
# ======================
class ModMLP(nn.Module):
    def __init__(self, p: int, d: int = 128, h: int = 512):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.fc1 = nn.Linear(2 * d, h)
        self.fc2 = nn.Linear(h, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ea = self.emb(x[:, 0])
        eb = self.emb(x[:, 1])
        z = torch.cat([ea, eb], dim=-1)
        z = F.relu(self.fc1(z))
        return self.fc2(z)


@torch.no_grad()
def eval_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
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
def run_one(
    seed,
    weight_decay,
    p=97,
    train_frac=0.3,
    epochs=10000,
    log_every=20,
    lr=1e-3,
    d=128,
    h=512,
):

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
    t95 = first_epoch_at_or_none(epochs_logged, test_acc_log, 0.95)

    return {
        "seed": seed,
        "weight_decay": weight_decay,
        "t_fit": t_fit,
        "t95": t95,
        "final_test_acc": test_acc_log[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive grok sweep with optional SPIR-V smoke test")
    parser.add_argument("--spv-demo", action="store_true", help="run a small Vulkan GEMV before training")
    parser.add_argument("--spv-size", type=int, default=64, help="vector size for the SPV GEMV demo")
    parser.add_argument("--device-index", type=int, default=0, help="Vulkan device index for the SPV demo")
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        help="torch device for training (e.g., cpu, cuda, cuda:0). Falls back to cpu if unavailable.",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="training epochs per run")
    parser.add_argument("--log-every", type=int, default=20, help="evaluation/print interval")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(3)), help="random seeds to sweep")
    parser.add_argument(
        "--coarse-wds",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0],
        help="coarse weight-decay values to scan",
    )
    parser.add_argument("--fine-radius", type=float, default=0.05, help="± range for fine scan around wd* that groks")
    parser.add_argument("--p", type=int, default=97, help="prime modulus")
    parser.add_argument("--train-frac", type=float, default=0.3, help="train/test split fraction")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--d", type=int, default=128, help="embedding dimension")
    parser.add_argument("--h", type=int, default=512, help="hidden width")
    parser.add_argument("--out", type=str, default="grok_sweep_adaptive_spv.csv", help="output CSV path")
    return parser.parse_args()


# ======================
# Adaptive Sweep
# ======================
def main():
    args = parse_args()

    # Select torch device once for all runs
    global device
    requested = args.torch_device
    if requested != "cpu" and not torch.cuda.is_available():
        print(f"[torch] Requested device '{requested}' unavailable; falling back to CPU.")
        device = "cpu"
    else:
        device = requested
    print(f"[torch] using device: {device}")

    if args.spv_demo:
        run_spv_demo(N=args.spv_size, device_index=args.device_index, verbose=True)

    seeds = args.seeds

    # ----- Phase 1: coarse scan -----
    coarse_wds = args.coarse_wds
    results = []

    print("\n--- Phase 1: Coarse Scan ---\n")

    for wd in coarse_wds:
        for seed in seeds:
            r = run_one(
                seed=seed,
                weight_decay=wd,
                p=args.p,
                train_frac=args.train_frac,
                epochs=args.epochs,
                log_every=args.log_every,
                lr=args.lr,
                d=args.d,
                h=args.h,
            )
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
    lower = max(0.0, wd_min - args.fine_radius)
    upper = wd_min + args.fine_radius
    fine_wds = np.round(np.arange(lower, upper, 0.01), 3).tolist()

    print("\n--- Phase 2: Fine Scan ---\n")

    for wd in fine_wds:
        for seed in seeds:
            r = run_one(
                seed=seed,
                weight_decay=wd,
                p=args.p,
                train_frac=args.train_frac,
                epochs=args.epochs,
                log_every=args.log_every,
                lr=args.lr,
                d=args.d,
                h=args.h,
            )
            results.append(r)
            print(f"wd={wd:<5} seed={seed:<2} t95={r['t95']} final_test={r['final_test_acc']:.3f}")

    # Save CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
