# ============================================
# runner_discrete_decoherence_avg.py
# ============================================
from __future__ import annotations
import argparse, numpy as np
import lattice_chsh as LCH
import chsh_extras  as CHEX
from scipy.linalg import expm

def main():
    p = argparse.ArgumentParser(description="Average S_fixed(k) over random phase seeds")
    p.add_argument("--NA", type=int, default=41)
    p.add_argument("--NB", type=int, default=41)
    p.add_argument("--M_A", type=int, default=6)
    p.add_argument("--M_B", type=int, default=9)
    p.add_argument("--t1", type=float, default=0.7)
    p.add_argument("--t2", type=float, default=1.3)
    p.add_argument("--kmin", type=int, default=6)
    p.add_argument("--kmax", type=int, default=24)
    p.add_argument("--kstep", type=int, default=3)
    p.add_argument("--tau", type=float, default=np.pi/8)
    p.add_argument("--Jprep", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--p_round", type=float, default=1.0)
    p.add_argument("--plot", action="store_true")

    args = p.parse_args()

    wallA, wallB = args.NA//2, args.NB//2
    sx, sy, sz = CHEX.pauli()
    H2 = CHEX.kron2(sx,sx)+CHEX.kron2(sy,sy)+CHEX.kron2(sz,sz)
    psi0 = np.zeros(4,complex); psi0[1]=1.0
    U = expm(-1j*args.Jprep*args.tau*H2)
    psi = U @ psi0
    a, ap, b, bp = 0.0, 0.5*np.pi, 0.25*np.pi, -0.25*np.pi
    # After computing psi and CHSH angles
    rho0 = CHEX.ket_to_rho(psi)
    p = 0.25  # any fixed p; depolarizing is k-independent
    S_dep = CHEX.S_rho(CHEX.depolarize_rho(rho0, p), a, ap, b, bp)



    print("k\t<S_fixed>\tσ(S)\tmean|uA| mean|uB|")

    k_list, mean_list, std_list = [], [], []

    for k in range(args.kmin, args.kmax+1, args.kstep):
        Svals=[]
        for s in range(args.seeds):
            rng=np.random.default_rng(s)
            HA, valsA, vecsA = LCH.build_single_leg_open_modulus_quantized_random(
                args.NA, args.t1, args.t2, wallA, args.M_A, k, p_round=args.p_round, rng=rng)
            HB, valsB, vecsB = LCH.build_single_leg_open_modulus_quantized_random(
                args.NB, args.t1, args.t2, wallB, args.M_B, k, p_round=args.p_round, rng=rng)

            uA,_=LCH.extract_wall_qubit_frame(vecsA,args.NA,wallA,which_block="A")
            uB,_=LCH.extract_wall_qubit_frame(vecsB,args.NB,wallB,which_block="A")
            WA,WB=CHEX.frame_unitary_from_basis(uA),CHEX.frame_unitary_from_basis(uB)
            Svals.append(CHEX.S_fixed_in_frames(psi,a,ap,b,bp,WA,WB))
        Svals=np.array(Svals)
        mean, std = Svals.mean(), Svals.std()
        k_list.append(k)
        mean_list.append(mean)
        std_list.append(std)
        print(f"{k}\t{mean:.6f}\t{std:.6f}\t1.0 1.0")

    # Optional plotting
    if args.plot:
        import matplotlib.pyplot as plt
        plt.errorbar(k_list, mean_list, yerr=std_list, marker='o', capsize=4)
        plt.axhline(2.0,  ls='--', label='Classical bound')
        plt.axhline(2.8284, ls=':', label='Tsirelson')
        plt.xlabel("Phase quantization k")
        plt.ylabel("<S_fixed> ± σ")
        plt.axhline(S_dep, ls='-.', label=f'Depolarizing (p={p:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def spectrum_of_series(y):
        y = np.asarray(y) - np.mean(y)
        Y = np.fft.rfft(y)          # real FFT
        f = np.fft.rfftfreq(len(y), d=1.0)  # cycles per k-step
        P = np.abs(Y)**2
        return f, P

    # If you ran with kstep=1, this highlights periodicities (look for ~1/18 cycles per k)
    if args.plot and args.kstep == 1:
        f, P = spectrum_of_series(mean_list)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(f, P, marker='o')
        plt.xlabel("cycles per k-step")
        plt.ylabel("power")
        plt.title("DFT of ⟨S_fixed⟩(k)")
        plt.tight_layout()
        plt.show()
 
if __name__=="__main__":
    main()
