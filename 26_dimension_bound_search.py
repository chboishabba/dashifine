#!/usr/bin/env python3
from __future__ import annotations
from itertools import product, permutations
from typing import List, Tuple, Dict, Set

Vec = Tuple[int, ...]  # coords in {-1,0,1}

def Q_sigma(x: Vec, sigma: Tuple[int, ...]) -> int:
    # sigma entries are +1 or -1
    return sum(s * (xi * xi) for s, xi in zip(sigma, x))

def signed_permutation_actions(p: int, q: int):
    """
    Group = (sign flips) ⋊ (permute within + block) × (permute within - block)
    We implement it explicitly for small dims.
    """
    m = p + q
    plus_idx = list(range(p))
    minus_idx = list(range(p, m))

    # permutations within blocks
    for perm_plus in permutations(plus_idx):
        for perm_minus in permutations(minus_idx):
            perm = list(range(m))
            for a, b in zip(plus_idx, perm_plus):
                perm[a] = b
            for a, b in zip(minus_idx, perm_minus):
                perm[a] = b

            # sign flips: independent flips on each coordinate
            for flips in product([-1, 1], repeat=m):
                def act(x: Vec, perm=tuple(perm), flips=tuple(flips)) -> Vec:
                    y = [0] * m
                    for i in range(m):
                        y[i] = flips[i] * x[perm[i]]
                    return tuple(y)
                yield act

def orbit_partition(points: List[Vec], actions) -> List[Set[Vec]]:
    remaining: Set[Vec] = set(points)
    orbits: List[Set[Vec]] = []
    actions = list(actions)  # materialize once
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

def search(max_m: int = 8, shell_value: int = 1):
    for m in range(2, max_m + 1):
        print(f"\n=== m={m} ===")
        for p in range(1, m):
            q = m - p
            # diagonal signature mask: first p plus, next q minus
            sigma = tuple([1]*p + [-1]*q)

            pts = trit_points(m)
            shell = [x for x in pts if abs(Q_sigma(x, sigma)) == shell_value]
            if not shell:
                continue

            actions = signed_permutation_actions(p, q)
            orbits = orbit_partition(shell, actions)

            sizes = sorted([len(o) for o in orbits], reverse=True)
            print(f"sig=({p},{q}) shell={len(shell)} orbits={len(orbits)} top_sizes={sizes[:5]}")

if __name__ == "__main__":
    search(max_m=8, shell_value=1)
