# ============================================
# runner_lattice_chsh.py — Run lattice-aware CHSH (executes only under __main__)
# ============================================

from __future__ import annotations
import argparse
import numpy as np
import lattice_chsh as LCH

def main():
    p = argparse.ArgumentParser(description="Lattice-aware CHSH using SSH wall modes")
    p.add_argument("--NA", type=int, default=41, help="cells for leg A (open)")
    p.add_argument("--NB", type=int, default=41, help="cells for leg B (open)")
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    p.add_argument("--prep", type=str, default="ideal_bell", choices=["ideal_bell","heisenberg"])
    p.add_argument("--Jprep", type=float, default=1.0, help="Heisenberg coupling for entangler")
    p.add_argument("--tau", type=float, default=np.pi/8, help="Heisenberg evolution time")
    p.add_argument("--blockA", type=str, default="A", choices=["A","B"], help="sublattice to read at wall for A")
    p.add_argument("--blockB", type=str, default="A", choices=["A","B"], help="sublattice to read at wall for B")
    p.add_argument("--angles", type=float, nargs=4, default=None, metavar=("a","ap","b","bp"),
                   help="custom CHSH angles (radians)")
    args = p.parse_args()

    res = LCH.chsh_on_lattice_frames(
        N_A=args.NA, N_B=args.NB,
        t1=args.t1, t2=args.t2,
        which_block_A=args.blockA,
        which_block_B=args.blockB,
        prep_mode=args.prep, Jprep=args.Jprep, tau=args.tau,
        angles=tuple(args.angles) if args.angles is not None else None,
    )

    print("\n=== Lattice-aware CHSH ===")
    print(f"prep_mode = {args.prep}  (Jprep={args.Jprep}, tau={args.tau})")
    a, ap, b, bp = res["angles"]
    print(f"Angles: a={a:.3f}, a'={ap:.3f}, b={b:.3f}, b'={bp:.3f}")
    print(f"S ≈ {res['S']:.6f}")

    print("\nNear-zero eigenvalues (first few) for leg A and B:")
    print("A:", res["valsA"])
    print("B:", res["valsB"])

    # Optional: print wall-plane basis summaries
    uA = res["wall_frames"]["uA"]; uB = res["wall_frames"]["uB"]
    print("\nWall frame norms (uA,uB):", np.linalg.norm(uA), np.linalg.norm(uB))

if __name__ == "__main__":
    main()
