#!/usr/bin/env python3
"""
40_generate_orbit_profiles.py

Generate orbit-profile tables for given signatures (p,q) and shells |Q|=k
under signed permutations within + and - blocks.

Outputs:
  - per-shell orbit-size CSVs:
      orbit_profile_p{p}_q{q}_shell{shell}.csv
  - shell-level summary table:
      orbit_shell_summary.csv
  - signature-level generating-function style summary:
      orbit_generating_functions.csv
"""

from __future__ import annotations

import argparse
from itertools import product, permutations
from pathlib import Path
from typing import Dict, List, Tuple, Set, Iterable
import csv

Vec = Tuple[int, ...]  # coords in {-1,0,1}


def Q_sigma(x: Vec, sigma: Tuple[int, ...]) -> int:
    return sum(s * (xi * xi) for s, xi in zip(sigma, x))


def signed_permutation_actions(p: int, q: int):
    """
    Group = (sign flips) ⋊ (permute within + block) × (permute within - block)
    """
    m = p + q
    plus_idx = list(range(p))
    minus_idx = list(range(p, m))

    for perm_plus in permutations(plus_idx):
        for perm_minus in permutations(minus_idx):
            perm = list(range(m))
            for a, b in zip(plus_idx, perm_plus):
                perm[a] = b
            for a, b in zip(minus_idx, perm_minus):
                perm[a] = b

            for flips in product([-1, 1], repeat=m):
                def act(x: Vec, perm=tuple(perm), flips=tuple(flips)) -> Vec:
                    y = [0] * m
                    for i in range(m):
                        y[i] = flips[i] * x[perm[i]]
                    return tuple(y)
                yield act


def orbit_partition(points: List[Vec], actions: Iterable) -> List[Set[Vec]]:
    remaining: Set[Vec] = set(points)
    actions = list(actions)
    orbits: List[Set[Vec]] = []
    while remaining:
        seed = next(iter(remaining))
        orb = set([seed])
        frontier = [seed]
        while frontier:
            cur = frontier.pop()
            for act in actions:
                nxt = act(cur)
                if nxt in remaining and nxt not in orb:
                    orb.add(nxt)
                    frontier.append(nxt)
        remaining -= orb
        orbits.append(orb)
    return orbits


def trit_points(m: int) -> List[Vec]:
    return [tuple(v) for v in product([-1, 0, 1], repeat=m)]


def write_sizes_csv(path: Path, sizes: List[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["size"])
        for s in sizes:
            w.writerow([s])


def write_rows_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_sigs(sig_args: List[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s in sig_args:
        parts = s.split(",")
        if len(parts) != 2:
            raise SystemExit(f"bad signature '{s}', expected p,q")
        p, q = int(parts[0]), int(parts[1])
        out.append((p, q))
    return out


def parse_shells(shell_args: List[str]) -> List[int]:
    return [int(x) for x in shell_args]


def poly_string(coeff_by_shell: Dict[int, int], var: str = "x") -> str:
    terms: List[str] = []
    for shell in sorted(coeff_by_shell):
        coeff = coeff_by_shell[shell]
        if shell == 0:
            terms.append(str(coeff))
        elif shell == 1:
            terms.append(f"{coeff}{var}")
        else:
            terms.append(f"{coeff}{var}^{shell}")
    return " + ".join(terms) if terms else "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigs", nargs="+", required=True, help="list of p,q signatures")
    ap.add_argument("--shells", nargs="+", required=True, help="list of shell values (|Q|=k)")
    ap.add_argument("--out-dir", default="orbit_profiles")
    args = ap.parse_args()

    sigs = parse_sigs(args.sigs)
    shells = parse_shells(args.shells)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shell_summary_rows: List[Dict[str, object]] = []
    generating_rows: List[Dict[str, object]] = []

    for p, q in sigs:
        m = p + q
        sigma = tuple([1] * p + [-1] * q)
        pts = trit_points(m)
        actions = list(signed_permutation_actions(p, q))
        shell_point_coeffs: Dict[int, int] = {}
        shell_orbit_coeffs: Dict[int, int] = {}
        signature_label = f"p{p}_q{q}"

        for shell in shells:
            shell_pts = [x for x in pts if abs(Q_sigma(x, sigma)) == shell]
            if not shell_pts:
                continue
            orbits = orbit_partition(shell_pts, actions)
            sizes = sorted([len(o) for o in orbits], reverse=True)
            out = out_dir / f"orbit_profile_p{p}_q{q}_shell{shell}.csv"
            write_sizes_csv(out, sizes)
            shell_point_coeffs[shell] = len(shell_pts)
            shell_orbit_coeffs[shell] = len(orbits)
            shell_summary_rows.append({
                "signature": signature_label,
                "p": p,
                "q": q,
                "shell": shell,
                "ambient_dim": m,
                "shell_point_count": len(shell_pts),
                "orbit_count": len(orbits),
                "largest_orbit": max(sizes),
                "smallest_orbit": min(sizes),
                "orbit_sizes": ";".join(str(s) for s in sizes),
            })
            print(f"p={p} q={q} shell={shell} -> {out} (orbits={len(orbits)} top={sizes[:5]})")

        generating_row: Dict[str, object] = {
            "signature": signature_label,
            "p": p,
            "q": q,
            "ambient_dim": m,
            "shell_point_total": sum(shell_point_coeffs.values()),
            "orbit_total": sum(shell_orbit_coeffs.values()),
            "shell_point_poly": poly_string(shell_point_coeffs),
            "orbit_count_poly": poly_string(shell_orbit_coeffs),
        }
        for shell in sorted(shells):
            generating_row[f"shell_{shell}_points"] = shell_point_coeffs.get(shell, 0)
            generating_row[f"shell_{shell}_orbits"] = shell_orbit_coeffs.get(shell, 0)
        generating_rows.append(generating_row)

    write_rows_csv(
        out_dir / "orbit_shell_summary.csv",
        [
            "signature",
            "p",
            "q",
            "shell",
            "ambient_dim",
            "shell_point_count",
            "orbit_count",
            "largest_orbit",
            "smallest_orbit",
            "orbit_sizes",
        ],
        shell_summary_rows,
    )

    generating_fields = [
        "signature",
        "p",
        "q",
        "ambient_dim",
        "shell_point_total",
        "orbit_total",
        "shell_point_poly",
        "orbit_count_poly",
    ]
    for shell in sorted(shells):
        generating_fields.append(f"shell_{shell}_points")
        generating_fields.append(f"shell_{shell}_orbits")
    write_rows_csv(out_dir / "orbit_generating_functions.csv", generating_fields, generating_rows)
    print(f"wrote summary: {out_dir / 'orbit_shell_summary.csv'}")
    print(f"wrote generating functions: {out_dir / 'orbit_generating_functions.csv'}")


if __name__ == "__main__":
    main()
