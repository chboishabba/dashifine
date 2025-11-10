# ============================================
# runner_chsh_experiments.py
# ============================================
from __future__ import annotations
import argparse, numpy as np
import chsh_harness as CHSH     # your earlier module
import chsh_extras  as CHEX

def main():
    p = argparse.ArgumentParser(description="CHSH experiments: no-signalling, scans, noise")
    p.add_argument("--mode", choices=["nosig","scan-tau","noise"], default="nosig")
    args = p.parse_args()

    # Tsirelson angles
    a, ap, b, bp = CHSH.tsirelson_angles()

    if args.mode == "nosig":
        # Use ideal |Φ+> for a sharp test
        psi = CHSH.bell_state_phi_plus()
        rho = CHEX.ket_to_rho(psi)
        diffs = CHEX.no_signalling_violation(rho, a, ap, b, bp)
        S = CHEX.S_rho(rho, a, ap, b, bp)
        print("S =", S)
        print("No-signalling diffs:", diffs)

    # elif args.mode == "scan-tau":
      #  taus, Svals = CHEX.scan_tau_heisenberg()
      #  for t, s in zip(taus, Svals):
      #      print(f"tau={t:.5f}  S={s:.6f}")

    # maximal CHSH value 
    # S max⁡ for each τ replace the scan-tau branch with:
    elif args.mode == "scan-tau":
        taus, Svals = CHEX.scan_tau_heisenberg_Smax()
        for t, s in zip(taus, Svals):
            print(f"tau={t:.5f}  S_max={s:.6f}")



    elif args.mode == "noise":
        p_list, Svals = CHEX.scan_noise_on_bell()
        for p, s in zip(p_list, Svals):
            print(f"p={p:.3f}  S={s:.6f}")

if __name__ == "__main__":
    main()
