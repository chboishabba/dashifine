"""Predeclared wave-KRR projection family for BIDI validation.

This extends the single `wave_krr.py` benchmark into a small declared family
*before* looking at the family results.  A model-family claim is supported only
if the same ordering survives every admitted projection task.

The family intentionally contains information-reducing coordinate projections;
they are stress tests, not expected wins.  Failure is useful BIDI information:
it identifies which task information the spectral advantage actually needs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from dashifine.kernels import dashifine_kernel
from newtest.wave_krr import (
    krr_predict,
    make_wave_field,
    mse,
    periodic_rbf_kernel,
    rbf_kernel,
)


@dataclass(frozen=True)
class WaveProjectionTask:
    name: str
    description: str


DECLARED_TASKS = (
    WaveProjectionTask("full_xy", "Original two-coordinate torus observation."),
    WaveProjectionTask("x_only", "Erase the second observed coordinate."),
    WaveProjectionTask("y_only", "Erase the first observed coordinate."),
)


@dataclass(frozen=True)
class TaskComparisonReceipt:
    task: str
    dashifine_best_mse: float
    rbf_best_mse: float
    periodic_rbf_best_mse: float
    dominates_rbf: bool
    dominates_periodic_rbf: bool


@dataclass(frozen=True)
class WaveFamilyReceipt:
    family_declared_before_evaluation: bool
    task_names: tuple[str, ...]
    comparisons: tuple[TaskComparisonReceipt, ...]
    dashifine_dominates_every_baseline_on_every_task: bool
    one_task_win_promoted_to_family_claim: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _project(name: str, X: np.ndarray) -> np.ndarray:
    out = np.array(X, copy=True)
    if name == "full_xy":
        return out
    if name == "x_only":
        out[:, 1] = 0.0
        return out
    if name == "y_only":
        out[:, 0] = 0.0
        return out
    raise ValueError(f"unknown projection task: {name}")


def _best_mse(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    kernel_factories: list[Callable[[], Callable]],
) -> float:
    values = []
    for factory in kernel_factories:
        pred = krr_predict(Xtr, ytr, Xte, factory(), lam=1e-2)
        values.append(mse(pred, yte))
    return float(min(values))


def run_declared_family(
    *,
    n_points: int = 2000,
    modes: int = 5,
    noise: float = 0.03,
    field_seed: int = 42,
    split_seed: int = 0,
    n_train: int = 200,
) -> WaveFamilyReceipt:
    X, y = make_wave_field(
        n_points=n_points,
        M=modes,
        noise=noise,
        seed=field_seed,
    )
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(X))
    tr, te = idx[:n_train], idx[n_train:]

    temperatures = (0.3, 0.7, 1.2, 2.0, 4.0)
    lengthscales = (0.3, 0.7, 1.2, 2.0, 4.0)
    comparisons: list[TaskComparisonReceipt] = []

    for task in DECLARED_TASKS:
        XP = _project(task.name, X)
        Xtr, ytr = XP[tr], y[tr]
        Xte, yte = XP[te], y[te]

        dash_mse = _best_mse(
            Xtr,
            ytr,
            Xte,
            yte,
            [
                lambda T=T: (
                    lambda A, B: dashifine_kernel(
                        A, B, k_max=4, temperature=T
                    )
                )
                for T in temperatures
            ],
        )
        rbf_mse = _best_mse(
            Xtr,
            ytr,
            Xte,
            yte,
            [
                lambda ell=ell: (
                    lambda A, B: rbf_kernel(A, B, lengthscale=ell)
                )
                for ell in lengthscales
            ],
        )
        periodic_mse = _best_mse(
            Xtr,
            ytr,
            Xte,
            yte,
            [
                lambda ell=ell: (
                    lambda A, B: periodic_rbf_kernel(A, B, lengthscale=ell)
                )
                for ell in lengthscales
            ],
        )
        comparisons.append(
            TaskComparisonReceipt(
                task=task.name,
                dashifine_best_mse=dash_mse,
                rbf_best_mse=rbf_mse,
                periodic_rbf_best_mse=periodic_mse,
                dominates_rbf=dash_mse < rbf_mse,
                dominates_periodic_rbf=dash_mse < periodic_mse,
            )
        )

    family_dominance = all(
        row.dominates_rbf and row.dominates_periodic_rbf
        for row in comparisons
    )
    return WaveFamilyReceipt(
        family_declared_before_evaluation=True,
        task_names=tuple(task.name for task in DECLARED_TASKS),
        comparisons=tuple(comparisons),
        dashifine_dominates_every_baseline_on_every_task=family_dominance,
    )


def main() -> None:
    receipt = run_declared_family()
    out = Path("outputs/wave_krr_family_receipt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(receipt.to_dict(), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
