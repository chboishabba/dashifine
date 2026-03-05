#!/usr/bin/env python3
"""
40_generate_orbit_profiles.py

Generate orbit-profile tables for given signatures (p,q) and shells |Q|=k
under signed permutations within + and - blocks.

Outputs CSV files with a single column: size
Filename: orbit_profile_p{p}_q{q}_shell{shell}.csv
"""

from __future__ import annotations

import argparse
from itertools import product, permutations
from pathlib import Path
from typing import List, Tuple, Set, Iterable
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

    for p, q in sigs:
        m = p + q
        sigma = tuple([1] * p + [-1] * q)
        pts = trit_points(m)
        actions = list(signed_permutation_actions(p, q))
        for shell in shells:
            shell_pts = [x for x in pts if abs(Q_sigma(x, sigma)) == shell]
            if not shell_pts:
                continue
            orbits = orbit_partition(shell_pts, actions)
            sizes = sorted([len(o) for o in orbits], reverse=True)
            out = out_dir / f"orbit_profile_p{p}_q{q}_shell{shell}.csv"
            write_sizes_csv(out, sizes)
            print(f"p={p} q={q} shell={shell} -> {out} (orbits={len(orbits)} top={sizes[:5]})")


if __name__ == "__main__":
    main()
