import numpy as np
import matplotlib.pyplot as plt

# =========================
# Target signal (same as yours)
# =========================
T = 40.0
Fs = 800
t = np.linspace(0, T, int(T*Fs), endpoint=False)

true_freqs = np.array([1.0, 2.6, 4.2])
true_amps  = np.array([1.2, 0.8, 0.6])
true_phis  = np.array([0.3, 1.7, -0.8])

env_true = 0.5 + 0.5*np.sin(2*np.pi*0.25*t)

y_target = env_true * (
    true_amps[0]*np.sin(2*np.pi*true_freqs[0]*t + true_phis[0]) +
    true_amps[1]*np.sin(2*np.pi*true_freqs[1]*t + true_phis[1]) +
    true_amps[2]*np.sin(2*np.pi*true_freqs[2]*t + true_phis[2])
)

# =========================
# Model:
# yhat = b + e(t) * sum_k (A_k sin(2π f_k t) + B_k cos(2π f_k t))
# envelope e(t) = e0 + sum_m (Ce_m sin(2π fe_m t) + De_m cos(2π fe_m t))
# =========================
rng = np.random.default_rng(0)

# Carrier atoms (start underfit; will lift)
freqs = [1.0, 2.]()
