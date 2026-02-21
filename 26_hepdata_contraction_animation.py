#!/usr/bin/env python3
"""
Animate contraction flow in coefficient space.

For each observable .npz bundle:
- fit best polynomial (GLS if cov provided)
- build coefficient vectors for each alpha in a sweep using:
    even projection + shrinkage (k^2)
- embed coefficient vectors into 2D via PCA
- produce an animated GIF showing the trajectory as alpha increases

Usage:
  python 26_hepdata_contraction_animation.py --inp hepdata_npz --out contraction_gifs
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def safe_log(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


@dataclass
class PolyFit:
    deg: int
    coeff: np.ndarray
    bic: float


def fit_poly_centered_logx(x: np.ndarray, y: np.ndarray, cov: Optional[np.ndarray], deg: int) -> PolyFit:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lx = safe_log(x)
    ly = safe_log(y)
    xc = lx - np.mean(lx)
    N = len(x)

    A = np.vstack([xc**k for k in range(deg + 1)]).T

    if cov is None:
        beta = np.linalg.lstsq(A, ly, rcond=None)[0]
        r = ly - A @ beta
        chi2 = float(r @ r)
    else:
        cov = 0.5 * (cov + cov.T)
        jitter = 1e-12 * np.trace(cov) / max(1, N)
        cov = cov + jitter * np.eye(N)
        L = np.linalg.cholesky(cov)
        Aw = np.linalg.solve(L, A)
        yw = np.linalg.solve(L, ly)
        beta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        r = ly - A @ beta
        wres = np.linalg.solve(L, r)
        chi2 = float(wres @ wres)

    k = deg + 1
    bic = chi2 + k * math.log(max(2, N))
    return PolyFit(deg=deg, coeff=beta, bic=bic)


def best_poly_fit(x, y, cov, degs=(1, 2, 3, 4)) -> PolyFit:
    fits = [fit_poly_centered_logx(x, y, cov, d) for d in degs]
    fits.sort(key=lambda f: f.bic)
    return fits[0]


def even_projection_coeff(coeff: np.ndarray) -> np.ndarray:
    c = np.array(coeff, float)
    c[1::2] = 0.0
    return c


def contract_representation(coeff: np.ndarray, alpha: float) -> np.ndarray:
    c = even_projection_coeff(coeff)
    out = c.copy()
    for k in range(len(out)):
        out[k] = out[k] / (1.0 + alpha * (k * k))
    return out


def pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def fig_to_rgb_array(fig):
    fig.canvas.draw()

    # Works across QtAgg / Agg backends
    buf = np.asarray(fig.canvas.buffer_rgba())
    # buffer_rgba gives RGBA; drop alpha channel
    return buf[..., :3].copy()


def make_gif(frames: List[np.ndarray], outpath: Path, fps: int = 4):
    try:
        import imageio.v2 as imageio
        imageio.mimsave(outpath, frames, duration=1.0 / max(1, fps))
        return
    except Exception:
        pass
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(outpath, save_all=True, append_images=imgs[1:], duration=int(1000 / max(1, fps)), loop=0)
    except Exception as e:
        raise SystemExit(f"Could not write gif (need imageio or pillow). Error: {e}")


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    d = np.load(path, allow_pickle=True)
    x = d["x"]
    y = d["y"]
    cov = d["cov"] if "cov" in d.files else None
    name = str(d["name"]) if "name" in d.files else path.stem
    return x, y, cov, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="directory containing hepdata .npz bundles")
    ap.add_argument("--out", default="contraction_gifs", help="output directory")
    ap.add_argument("--alpha-sweep",
                    default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000",
                    help="comma-separated alphas")
    ap.add_argument("--fps", type=int, default=4, help="gif fps")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    alphas = [float(s) for s in args.alpha_sweep.split(",") if s.strip()]
    npz_files = sorted(inp.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz found in {inp}")

    for f in npz_files:
        x, y, cov, name = load_npz(f)
        pf = best_poly_fit(x, y, cov, degs=(1, 2, 3, 4))

        coeffs = []
        odd_even = []
        norms = []

        for a in alphas:
            c = contract_representation(pf.coeff, a)
            coeffs.append(c)
            odd = float(np.sum(np.abs(c[1::2])) / (np.sum(np.abs(c[0::2])) + 1e-30))
            odd_even.append(odd)
            norms.append(float(np.linalg.norm(c)))

        C = np.vstack(coeffs)
        Z = pca2(C)

        frames = []
        for i, a in enumerate(alphas):
            fig = plt.figure(figsize=(6, 5))
            plt.plot(Z[:i + 1, 0], Z[:i + 1, 1], marker="o")
            plt.scatter(Z[i, 0], Z[i, 1], s=120)
            plt.title(
                f"{name}: coeff-space contraction\n"
                f"alpha={a:.1e}  odd/even={odd_even[i]:.3g}  ||c||={norms[i]:.3g}"
            )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            frames.append(fig_to_rgb_array(fig))
            plt.close(fig)

        gif_path = out / f"{name}_coeff_contraction.gif"
        make_gif(frames, gif_path, fps=args.fps)

        np.savetxt(
            out / f"{name}_coeff_contraction_pca2.csv",
            np.column_stack([alphas, Z, odd_even, norms]),
            delimiter=",",
            header="alpha,pc1,pc2,odd_even,norm",
            comments="",
        )

        print(f"Wrote {gif_path}")

    print(f"Done. Output in: {out.resolve()}")


if __name__ == "__main__":
    main()
