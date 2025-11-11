# runner_hydrogenic_demo.py
from __future__ import annotations
import numpy as np
from hydrogenic import hydrogenic_spectrum

def main():
    for Z in [1, 2, 5, 10]:
        E, _, _ = hydrogenic_spectrum(Z=Z, l=0, nr=600, rmax=150.0, n_eigs=4)
        print(f"Z={Z:2d}  E[:4] = {np.array2string(E, precision=6)}")

if __name__=="__main__":
    main()
