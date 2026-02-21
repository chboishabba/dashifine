import numpy as np

def norm(x): return np.linalg.norm(x)

def energy_to_proto(x, p):  # squared distance
    d = x - p
    return float(d @ d)

class MultiBasinContraction:
    """
    Piecewise-linear contraction with multiple fixed points (prototypes).
    Motif = basin index.
    Optional hysteresis margin makes motif switching rare / controlled.
    """
    def __init__(self, prototypes, alpha=0.75, hysteresis=0.0):
        self.P = np.array(prototypes, dtype=float)  # shape (K, d)
        self.alpha = float(alpha)
        assert 0.0 < self.alpha < 1.0
        self.hys = float(hysteresis)
        self.K, self.d = self.P.shape

    def assign(self, x, prev=None):
        # energies to each prototype
        E = np.sum((self.P - x[None, :])**2, axis=1)  # (K,)
        k_best = int(np.argmin(E))

        if prev is None or self.hys <= 0.0:
            return k_best

        # hysteresis: only switch if best beats prev by margin
        if E[k_best] + self.hys < E[int(prev)]:
            return k_best
        return int(prev)

    def step(self, x, motif=None):
        x = np.array(x, dtype=float)
        m = self.assign(x, prev=motif)
        p = self.P[m]
        x_next = p + self.alpha * (x - p)
        return x_next, m

    # Coarse Lyapunov: distance to assigned prototype
    def V(self, x, motif=None):
        m = self.assign(np.array(x, dtype=float), prev=motif)
        return energy_to_proto(x, self.P[m]), m


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    d = 9
    K = 9  # motifs M1..M9
    prototypes = rng.normal(size=(K, d))
    sys = MultiBasinContraction(prototypes, alpha=0.75, hysteresis=0.05)

    x = rng.normal(size=d) * 3.0
    motif = None

    print("n   motif   V(x)")
    for n in range(20):
        Vx, motif = sys.V(x, motif)
        print(f"{n:02d}   {motif+1:02d}    {Vx:.6f}")
        x, motif = sys.step(x, motif)
