#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import random

Vec = List[int]
Mat = List[List[int]]

def vadd(a: Vec, b: Vec) -> Vec:
    return [x + y for x, y in zip(a, b)]

def dot(a: Vec, b: Vec) -> int:
    return sum(x * y for x, y in zip(a, b))

def matvec(G: Mat, x: Vec) -> Vec:
    return [dot(row, x) for row in G]

def bilinear_from_gram(G: Mat, x: Vec, y: Vec) -> int:
    return dot(x, matvec(G, y))

def basis(n: int, i: int) -> Vec:
    e = [0] * n
    e[i] = 1
    return e

def derive_gram2(Q: Callable[[Vec], int], n: int) -> Mat:
    """
    Returns G2 such that for all x,y:
      B2(x,y) = Q(x+y) - Q(x) - Q(y) = x^T G2 y
    Over ℤ, G2 will have even diagonals if Q is 'honestly quadratic'.
    """
    G2: Mat = [[0] * n for _ in range(n)]
    for i in range(n):
        ei = basis(n, i)
        # B2(ei,ei) = Q(2ei) - 2Q(ei) = 2 * (quadratic coeff on i)
        # But easiest is to fill G2 via polarization on basis sums:
        for j in range(n):
            ej = basis(n, j)
            # B2(ei,ej)
            bij = Q(vadd(ei, ej)) - Q(ei) - Q(ej)
            G2[i][j] = bij
    return G2

def B2(Q: Callable[[Vec], int], x: Vec, y: Vec) -> int:
    return Q(vadd(x, y)) - Q(x) - Q(y)

# --- Your DASHI core quadratic: Qcore = sum of squares of {-1,0,1} coords.
def Qcore_int(x: Vec) -> int:
    return sum(t*t for t in x)

def random_trit_vec(n: int) -> Vec:
    return [random.choice([-1, 0, 1]) for _ in range(n)]

def main():
    random.seed(0)

    n = 6  # core length (m)
    Q = Qcore_int

    G2 = derive_gram2(Q, n)
    print("Derived G2 (represents B2 = 2B):")
    for row in G2:
        print(row)

    # Sanity check: B2(x,y) == x^T G2 y on random trit vectors
    for k in range(20):
        x = random_trit_vec(n)
        y = random_trit_vec(n)
        lhs = B2(Q, x, y)
        rhs = bilinear_from_gram(G2, x, y)
        if lhs != rhs:
            print("Mismatch!")
            print("x=", x)
            print("y=", y)
            print("B2(Q,x,y)=", lhs)
            print("x^T G2 y =", rhs)
            return

    print("OK: polarization matches x^T G2 y on random tests.")

    # For Qcore, expect G2 = 2I, so B(x,y)=dot(x,y)
    # We can demonstrate: B2(x,y)=2*dot(x,y)
    for k in range(5):
        x = random_trit_vec(n)
        y = random_trit_vec(n)
        print("x,y:", x, y, "B2:", B2(Q, x, y), "2dot:", 2*dot(x,y))

if __name__ == "__main__":
    main()
