import glob, os
import numpy as np
import pandas as pd

def load_beta_trajs(root="hepdata_dashi_native", d=5):
    paths = sorted(glob.glob(os.path.join(root, "*_dashi_native_metrics.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_dashi_native_metrics.csv under {root}")
    obs = [os.path.basename(p).replace("_dashi_native_metrics.csv", "") for p in paths]
    dfs = [pd.read_csv(p).sort_values("iter") for p in paths]
    X = np.stack([df[[f"b{k}" for k in range(d)]].to_numpy(float) for df in dfs], axis=0)
    # X: (Nobs, T, d)
    return obs, X

def E_L2(a,b): return float(np.sum((a-b)**2))
def E_L1(a,b): return float(np.sum(np.abs(a-b)))

def fejer_to_final(X, E):
    Nobs,T,d = X.shape
    final = X[:, -1, :]
    fracs = []
    for i in range(Nobs):
        ok = 0
        for t in range(T-1):
            if E(X[i,t+1,:], final[i]) <= E(X[i,t,:], final[i]) + 1e-12:
                ok += 1
        fracs.append(ok/(T-1))
    return np.array(fracs)

def parallelogram_residual(X):
    # Uses L2-norm² as Q; returns residual stats per iter.
    Nobs,T,d = X.shape
    rng = np.random.default_rng(0)
    res = []
    for t in range(T):
        # pick random pairs of observables at same iter
        vals = []
        for _ in range(min(200, Nobs*(Nobs-1)//2)):
            i,j = rng.integers(0, Nobs, size=2)
            if i==j: continue
            x = X[i,t,:]; y = X[j,t,:]
            lhs = np.sum((x+y)**2) + np.sum((x-y)**2)
            rhs = 2*np.sum(x**2) + 2*np.sum(y**2)
            vals.append(lhs-rhs)
        vals = np.array(vals, float) if vals else np.array([0.0])
        res.append((np.mean(vals), np.median(vals), np.quantile(np.abs(vals), 0.9)))
    return np.array(res)  # shape (T,3): mean, median, 90%|Δ|

def trit_quantize(v, eps):
    out = np.zeros_like(v, dtype=np.int8)
    out[v > eps] = 1
    out[v < -eps] = -1
    return out

def agree_depth(a, b):
    # a,b are 1D int arrays same length; LCP length
    k = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            k += 1
        else:
            break
    return k

def dNatFine(a, b):
    # mimic “distance = n - agreeDepth(reverse a, reverse b)”
    ra = a[::-1]; rb = b[::-1]
    return len(a) - agree_depth(ra, rb)

def cone_premise(a, b, eps=0.2):
    dv = trit_quantize(b-a, eps)
    pos = int((dv==1).sum())
    neg = int((dv==-1).sum())
    return (pos >= neg)

def cone_monotonicity_test(X, eps=0.2, qeps=0.15):
    # Quantize beta -> trits, then check: if in-cone at t, remains in-cone at t+1.
    Nobs,T,d = X.shape
    premises=0; viol=0
    examples=[]
    for t in range(T-1):
        P = X[:,t,:]
        Q = X[:,t+1,:]
        for i in range(Nobs):
            for j in range(Nobs):
                if i==j: continue
                if not cone_premise(P[i], P[j], eps=eps):
                    continue
                premises += 1
                if not cone_premise(Q[i], Q[j], eps=eps):
                    viol += 1
                    if len(examples) < 10:
                        ai = trit_quantize(P[i], qeps); aj = trit_quantize(P[j], qeps)
                        bi = trit_quantize(Q[i], qeps); bj = trit_quantize(Q[j], qeps)
                        examples.append({
                            "t": t,
                            "pair": (i,j),
                            "dNatFine_t": dNatFine(ai, aj),
                            "dNatFine_t1": dNatFine(bi, bj),
                            "ai": ai.tolist(), "aj": aj.tolist(),
                            "bi": bi.tolist(), "bj": bj.tolist(),
                        })
    return premises, viol, examples

if __name__ == "__main__":
    obs, X = load_beta_trajs("hepdata_dashi_native", d=5)

    f_l2 = fejer_to_final(X, E_L2)
    f_l1 = fejer_to_final(X, E_L1)
    print("Fejér-to-final fractions per observable:")
    for name, a, b in zip(obs, f_l2, f_l1):
        print(f"  {name:20s}  L2:{a:0.3f}  L1:{b:0.3f}")
    print("  mean L2:", float(np.mean(f_l2)), "mean L1:", float(np.mean(f_l1)))

    pr = parallelogram_residual(X)
    print("\nParallelogram residual over iterations (mean, median, 90%|Δ|):")
    for t,(m,md,q) in enumerate(pr):
        print(f"  iter {t:3d}: mean {m: .3e}  median {md: .3e}  q90|Δ| {q: .3e}")

    premises, viol, ex = cone_monotonicity_test(X, eps=0.2, qeps=0.15)
    print(f"\nCone-monotonicity premise count: {premises}, violations: {viol}, violation rate: {0 if premises==0 else viol/premises:0.4f}")
    if ex:
        print("Examples (first few):")
        for e in ex:
            print(e)
