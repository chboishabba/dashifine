# ============================================
# runner_triality_chsh.py
# ============================================
# Usage:
#   python runner_triality_chsh.py --plot
#
# What it does:
#   1) Builds a three-leg "triality" SSH stack (open chains, one wall per leg),
#      with gauge-covariant links and inter-leg couplings carrying 120° phases.
#      It scans g_perp values and prints the three smallest |E| (the split triplet).
#      With --plot, it draws the near-zero spectrum vs g_perp.
#
#   2) Evaluates a CHSH/Bell test on an ideal 2-qubit Bell state (|Φ+>),
#      using Tsirelson angles (expect S ≈ 2√2).
#
#   (Optional stubs/comments at bottom show how to map your lattice wall-mode
#    planes to qubit measurement axes for a lattice-aware CHSH experiment.)
#
# Requirements:
#   - numpy  (required)
#   - matplotlib (only if you pass --plot)
#
# NOTE: This file DOES NOT execute anything on import.
#       Code runs only under `if __name__ == "__main__":`

from __future__ import annotations
import argparse
import numpy as np
import numpy.linalg as LA
from math import pi

# ---- import the two helper modules you pasted earlier ----
# (Ensure triality_stack.py and chsh_harness.py are in the same directory)
import triality_stack as TST
import chsh_harness as CHSH


def build_phases_for_three_legs(N: int, wall: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Make three leg-specific phase programs:
      - Intra phases = 0
      - Inter phases = mod-6 bump + 120° offsets for legs 0/1/2.
    """
    offsets = (0.0, 2 * np.pi / 3.0, 4 * np.pi / 3.0)
    phases_leg = []
    for off in offsets:
        ti, tj = TST.make_leg_phases_with_offset(N, wall, off)
        phases_leg.append((ti, tj))
    return phases_leg


def scan_triality_triplet(
    N: int = 27,
    t1: float = 0.7,
    t2: float = 1.3,
    g_list = None,
    wall: int | None = None,
) -> dict:
    """
    Build the three-leg stack and scan g_perp to see the near-zero 'triality triplet' splitting.
    Returns a dict with eigen-snapshots.
    """
    if wall is None:
        wall = N // 2
    if g_list is None:
        g_list = np.linspace(0.0, 0.15, 11)

    phases_leg = build_phases_for_three_legs(N, wall)
    leg_dim = 4 * N

    triplet_spectra = []  # three smallest |E| per g
    details = []

    for g in g_list:
        H = TST.build_triality_stack_H(
            N=N, t1=t1, t2=t2,
            phases_leg=phases_leg,
            domain_wall_at=wall,
            g_perp=g,
            interleg_phase_shifts=(0.0, 2 * pi / 3, 4 * pi / 3),
        )
        vals, vecs = TST.eigh_sorted_by_abs(H)
        near = vals[:8]  # first few near zero
        triplet = vals[:3]
        triplet_spectra.append(triplet)

        # leg weights of the lowest mode (sanity peek)
        v0 = vecs[:, 0]
        w = []
        for ell in range(3):
            seg = v0[ell * leg_dim : (ell + 1) * leg_dim]
            w.append(float(np.dot(seg, seg)))
        w = np.array(w) / np.sum(w)
        details.append(dict(g=g, near=near, lowest_leg_weights=w))

    return {
        "N": N, "t1": t1, "t2": t2, "wall": wall,
        "g_list": np.asarray(g_list),
        "triplet_spectra": np.asarray(triplet_spectra),
        "details": details,
    }


def maybe_plot_triality(results: dict, show: bool = True, savepath: str | None = None):
    """
    Plot the three smallest |E| vs g_perp, if matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] matplotlib not available:", e)
        return

    g = results["g_list"]
    tri = results["triplet_spectra"]  # shape: [len(g), 3]

    plt.figure()
    for i in range(3):
        plt.plot(g, tri[:, i], marker='o', label=f"mode {i}")
    plt.title("Three lowest |E| vs inter-leg coupling $g_\\perp$ (triality stack)")
    plt.xlabel("$g_\\perp$")
    plt.ylabel("|E|")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
        print(f"[plot] saved to {savepath}")
    if show:
        plt.show()


def run_chsh_demo():
    """
    Evaluate the CHSH S-parameter on an ideal Bell state |Φ+> with Tsirelson angles.
    Expect S ≈ 2*sqrt(2).
    """
    a, ap, b, bp = CHSH.tsirelson_angles()
    psi = CHSH.bell_state_phi_plus()     # or CHSH.bell_state_psi_plus()
    S = CHSH.chsh_S(psi, a, ap, b, bp)
    return dict(S=S, angles=(a, ap, b, bp))


def main():
    parser = argparse.ArgumentParser(description="Three-leg triality + CHSH runner")
    parser.add_argument("--N", type=int, default=27, help="Cells per leg (open chain)")
    parser.add_argument("--t1", type=float, default=0.7, help="SSH weak bond")
    parser.add_argument("--t2", type=float, default=1.3, help="SSH strong bond")
    parser.add_argument("--gmin", type=float, default=0.0, help="g_perp min")
    parser.add_argument("--gmax", type=float, default=0.15, help="g_perp max")
    parser.add_argument("--gsteps", type=int, default=11, help="number of g_perp samples")
    parser.add_argument("--plot", action="store_true", help="Enable plots (matplotlib)")
    parser.add_argument("--savefig", type=str, default="", help="Optional path to save the triplet plot")
    args = parser.parse_args()

    # ----- Part 1: Triality stack scan -----
    g_list = np.linspace(args.gmin, args.gmax, args.gsteps)
    triality = scan_triality_triplet(
        N=args.N, t1=args.t1, t2=args.t2, g_list=g_list, wall=args.N // 2
    )
    print("\n=== Triality stack: three smallest |E| vs g_perp ===")
    for d in triality["details"]:
        g = d["g"]
        near = d["near"]
        w = d["lowest_leg_weights"]
        print(f"g_perp={g:6.3f}  near-zero eigs: {near}  lowest-mode leg-weights: {w}")

    if args.plot:
        savepath = args.savefig if args.savefig else None
        maybe_plot_triality(triality, show=True, savepath=savepath)

    # ----- Part 2: CHSH on an ideal Bell state -----
    bell = run_chsh_demo()
    a, ap, b, bp = bell["angles"]
    print("\n=== CHSH (ideal |Φ+>, Tsirelson angles) ===")
    print(f"S ≈ {bell['S']:.6f}   angles: a={a:.3f}, a'={ap:.3f}, b={b:.3f}, b'={bp:.3f}")
    print("Expected ~ 2*sqrt(2) ≈ 2.828427")

    # ---------------------------
    # Optional (comments only): lattice-aware CHSH mapping steps
    # ---------------------------
    #
    # If you want to use actual lattice wall-mode directions as local measurement frames:
    #
    # 1) Build TWO independent single-leg systems (A and B), each with an open SSH wall:
    #    T_A = TST.build_T_open(...); H_A = TST.build_chiral_H_from_T(T_A)
    #    T_B = TST.build_T_open(...); H_B = TST.build_chiral_H_from_T(T_B)
    #
    # 2) Extract their near-zero eigenvectors, then pull the local 2D (cos,sin) direction at the wall:
    #    valsA, vecsA = TST.eigh_sorted_by_abs(H_A)
    #    vA = vecsA[:, 0]    # near-zero
    #    uA = CHSH.extract_local_plane_basis_at_wall(vA, N=NA, wall_cell=wallA, which_block="A")
    #
    #    valsB, vecsB = TST.eigh_sorted_by_abs(H_B)
    #    vB = vecsB[:, 0]
    #    uB = CHSH.extract_local_plane_basis_at_wall(vB, N=NB, wall_cell=wallB, which_block="A")
    #
    # 3) Build a Bell state in those local bases:
    #    psi_loc = CHSH.two_qubit_from_two_local_planes(uA, uB)
    #
    # 4) Evaluate CHSH the same way:
    #    a, ap, b, bp = CHSH.tsirelson_angles()
    #    S_loc = CHSH.chsh_S(psi_loc, a, ap, b, bp)
    #    print("S (local frames) =", S_loc)
    #
    # If you prefer to simulate an actual entangling PREP between two wall modes,
    # you can build a 4x4 effective Hamiltonian in the wall-mode basis and evolve
    # a product state briefly to create an entangled state, then evaluate S.


if __name__ == "__main__":
    main()
