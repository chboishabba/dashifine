import numpy as np

J = np.load("operator_analysis_v2/J_global.npy")
d = J.shape[0]

def sig(A, tol=1e-8):
    A = 0.5*(A + A.T)  # ensure symmetric for eigen-signature
    w = np.linalg.eigvalsh(A)
    pos = int(np.sum(w > tol))
    neg = int(np.sum(w < -tol))
    zer = d - pos - neg
    return w, {"positive": pos, "negative": neg, "zero": zer}

# 1) Symmetric part of J
wS, sS = sig(0.5*(J + J.T), tol=1e-8)

# 2) Contraction certificate forms
wQ1, sQ1 = sig(np.eye(d) - J.T@J, tol=1e-8)
wQ2, sQ2 = sig(J.T@J - np.eye(d), tol=1e-8)

print("||J||2 approx via SVD:", np.linalg.svd(J, compute_uv=False)[0])
print("eig(J) (real parts):", np.real(np.linalg.eigvals(J)))
print("sig((J+J.T)/2):", sS, "eig:", wS)
print("sig(I - J.T J):", sQ1, "eig:", wQ1)
print("sig(J.T J - I):", sQ2, "eig:", wQ2)
