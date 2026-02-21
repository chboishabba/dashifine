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
freqs = [1.0, 2.0, 4.0]
A = list(rng.normal(0.0, 0.12, size=len(freqs)))
B = list(rng.normal(0.0, 0.12, size=len(freqs)))

# Envelope atoms (slow only)
env_freqs = [0.25]  # include the true one; you can start with [0.2] to test discovery
Ce = list(rng.normal(0.0, 0.10, size=len(env_freqs)))
De = list(rng.normal(0.0, 0.10, size=len(env_freqs)))
e0 = 0.5

# Affine bias
b = 0.0

# Mirror/reference pairing among carriers (indices)
# Pair (0,1), (2,3), ... as they exist.
def carrier_pairs(n):
    pairs = []
    for i in range(0, n-1, 2):
        pairs.append((i, i+1))
    return pairs

# =========================
# Hyperparameters
# =========================
lr_A = 5e-3
lr_B = 5e-3
lr_f = 3e-6

lr_Ce = 3e-3
lr_De = 3e-3
lr_e0 = 1e-3
lr_b  = 1e-3

repel = 1e-4          # soft frequency repulsion
mirror_k = 3e-3       # mirror/reference coupling (A,B pairs)
merge_every = 400
merge_delta = 0.12

steps = 14000
batch = 1600
snap_every = 25

lift_threshold = 0.05   # tighter target now that envelope exists
max_oscillators = 8
delta_f = 0.18
nudge_eta = 0.35

# =========================
# Helpers
# =========================
def envelope(tseg, e0, Ce, De, env_freqs):
    e = np.full_like(tseg, float(e0))
    for m, fe in enumerate(env_freqs):
        arg = 2*np.pi*fe*tseg
        e += Ce[m]*np.sin(arg) + De[m]*np.cos(arg)
    return e

def carrier_sum(tseg, A, B, freqs):
    s = np.zeros_like(tseg)
    for k, fk in enumerate(freqs):
        arg = 2*np.pi*fk*tseg
        s += A[k]*np.sin(arg) + B[k]*np.cos(arg)
    return s

def synth_full(tfull, b, e0, Ce, De, env_freqs, A, B, freqs):
    e = envelope(tfull, e0, Ce, De, env_freqs)
    c = carrier_sum(tfull, A, B, freqs)
    return b + e*c

def mags(A, B):
    A = np.asarray(A); B = np.asarray(B)
    return np.sqrt(A*A + B*B)

def dominant_residual_peak(residual, Fs, fmin=0.5, fmax=10.0):
    fft_vals = np.fft.rfft(residual)
    fft_freq = np.fft.rfftfreq(len(residual), 1/Fs)
    mask = (fft_freq >= fmin) & (fft_freq <= fmax)
    mags_ = np.abs(fft_vals[mask])
    if mags_.size == 0:
        return None, None
    i = int(np.argmax(mags_))
    return float(fft_freq[mask][i]), float(mags_[i])

def merge_nearby(freqs, A, B, delta):
    if len(freqs) <= 1:
        return freqs, A, B
    order = np.argsort(freqs)
    f = [float(freqs[i]) for i in order]
    a = [float(A[i]) for i in order]
    b = [float(B[i]) for i in order]
    out_f, out_A, out_B = [], [], []
    i = 0
    while i < len(f):
        f0, a0, b0 = f[i], a[i], b[i]
        j = i + 1
        while j < len(f) and abs(f[j] - f0) <= delta:
            a0 += a[j]
            b0 += b[j]
            w0 = abs(a0) + abs(b0) + 1e-12
            wj = abs(a[j]) + abs(b[j]) + 1e-12
            f0 = (w0*f0 + wj*f[j])/(w0+wj)
            j += 1
        out_f.append(float(f0))
        out_A.append(float(a0))
        out_B.append(float(b0))
        i = j
    return out_f, out_A, out_B

# =========================
# Logging
# =========================
E_hist = []
freq_hist = []
mag_hist = []
peak_hist = []
env_hist = []

lift_events = []

# =========================
# Training loop
# =========================
for it in range(steps):
    N = len(freqs)
    M = len(env_freqs)

    start = rng.integers(0, len(t)-batch)
    idx = slice(start, start+batch)
    tb = t[idx]
    yb = y_target[idx]

    # Forward pieces
    e = envelope(tb, e0, Ce, De, env_freqs)
    c = np.zeros_like(tb)

    sin_car = []
    cos_car = []
    for k in range(N):
        arg = 2*np.pi*freqs[k]*tb
        s = np.sin(arg)
        c0 = np.cos(arg)
        sin_car.append(s)
        cos_car.append(c0)
        c += A[k]*s + B[k]*c0
    sin_car = np.array(sin_car)
    cos_car = np.array(cos_car)

    # prediction
    y_hat = b + e*c
    err = y_hat - yb

    # ---- grads for affine b
    grad_b = 2*np.mean(err)

    # ---- grads for envelope params
    # y = b + e*c
    # d/d e0: c
    grad_e0 = 2*np.mean(err * c)

    grad_Ce = np.zeros(M)
    grad_De = np.zeros(M)
    for m, fe in enumerate(env_freqs):
        arg = 2*np.pi*fe*tb
        sm = np.sin(arg)
        cm = np.cos(arg)
        grad_Ce[m] = 2*np.mean(err * (sm * c))
        grad_De[m] = 2*np.mean(err * (cm * c))

    # ---- grads for carrier A/B
    # y = b + e * sum_k(A s + B c)
    grad_A = 2*np.mean(err[None,:] * (e[None,:] * sin_car), axis=1)
    grad_B = 2*np.mean(err[None,:] * (e[None,:] * cos_car), axis=1)

    # ---- freq grads (stabilized time for gradient only)
    tau = tb - tb.mean()
    tau = tau / (tau.std() + 1e-9)

    grad_f = np.zeros(N)
    for k in range(N):
        # d/d f of (A sin + B cos) = (A cos - B sin) * 2π t
        term = (A[k]*cos_car[k] - B[k]*sin_car[k]) * (2*np.pi * tau)
        grad_f[k] = 2*np.mean(err * (e * term))

    # ---- mirror/reference coupling: encourage paired (A,B) to be similar or negated
    # Here we encourage similarity (you can flip sign to encourage anti-alignment).
    for (i, j) in carrier_pairs(N):
        grad_A[i] += mirror_k * (A[i] - A[j])
        grad_A[j] += mirror_k * (A[j] - A[i])
        grad_B[i] += mirror_k * (B[i] - B[j])
        grad_B[j] += mirror_k * (B[j] - B[i])

    # ---- frequency repulsion (soft)
    for k in range(N):
        rep = 0.0
        for j in range(N):
            if j == k:
                continue
            rep += 1.0 / ((freqs[k] - freqs[j])**2 + 0.05)
        grad_f[k] += repel * rep

    # ---- updates
    b  -= lr_b * grad_b
    e0 -= lr_e0 * grad_e0
    for m in range(M):
        Ce[m] -= lr_Ce * grad_Ce[m]
        De[m] -= lr_De * grad_De[m]

    for k in range(N):
        A[k] -= lr_A * grad_A[k]
        B[k] -= lr_B * grad_B[k]
        freqs[k] -= lr_f * grad_f[k]

    # gauge-fix duplicates occasionally
    if (it + 1) % merge_every == 0:
        freqs, A, B = merge_nearby(freqs, A, B, merge_delta)

    # ---- snapshot + obstruction/lift
    if it % snap_every == 0:
        y_full = synth_full(t, b, e0, Ce, De, env_freqs, A, B, freqs)
        mse = float(np.mean((y_full - y_target)**2))

        E_hist.append(mse)
        freq_hist.append(freqs.copy())
        mag_hist.append(mags(A, B).tolist())
        env_hist.append((float(e0), Ce.copy(), De.copy()))

        peak_f, peak_mag = dominant_residual_peak(y_target - y_full, Fs, fmin=0.5, fmax=10.0)
        peak_hist.append(peak_f if peak_f is not None else np.nan)

        if (peak_f is not None) and (mse > lift_threshold) and (len(freqs) < max_oscillators):
            diffs = np.array([abs(peak_f - f) for f in freqs])
            k_near = int(np.argmin(diffs))
            if diffs[k_near] <= delta_f:
                freqs[k_near] = (1.0 - nudge_eta) * freqs[k_near] + nudge_eta * peak_f
                lift_events.append((len(E_hist)-1, peak_f, "NUDGE"))
            else:
                freqs.append(float(peak_f))
                A.append(0.0); B.append(0.0)
                lift_events.append((len(E_hist)-1, peak_f, "ADD"))

# =========================
# Final synthesis & plots
# =========================
E_hist = np.array(E_hist)
y_learned = synth_full(t, b, e0, Ce, De, env_freqs, A, B, freqs)

# Plot 1: same main graph (target vs learned)
plt.figure()
win = slice(0, 2000)
plt.plot(t[win], y_target[win], label="Target")
plt.plot(t[win], y_learned[win], label="Learned")
plt.title("Adaptive Fit with Explicit Coupled Envelope (affine + mirror + lift)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot 2: MSE
plt.figure()
plt.plot(E_hist)
plt.title("Energy (MSE) During Training (with explicit envelope)")
plt.xlabel("Training snapshot")
plt.ylabel("MSE")
plt.show()

# Plot 3: frequency trajectories
max_len = max(len(f) for f in freq_hist)
freq_matrix = np.full((len(freq_hist), max_len), np.nan)
for i, f in enumerate(freq_hist):
    freq_matrix[i, :len(f)] = f

plt.figure()
for k in range(freq_matrix.shape[1]):
    plt.plot(freq_matrix[:, k])
plt.title("Carrier Frequency Trajectories")
plt.xlabel("Training snapshot")
plt.ylabel("Frequency (Hz)")
plt.show()

# Plot 4: magnitudes
mag_matrix = np.full((len(mag_hist), max_len), np.nan)
for i, m in enumerate(mag_hist):
    mag_matrix[i, :len(m)] = m

plt.figure()
for k in range(mag_matrix.shape[1]):
    plt.plot(mag_matrix[:, k])
plt.title("Carrier Component Magnitudes (internal strengths)")
plt.xlabel("Training snapshot")
plt.ylabel("Magnitude")
plt.show()

# Plot 5: residual spectrum
residual = y_target - y_learned
fft_vals = np.fft.rfft(residual)
fft_freq = np.fft.rfftfreq(len(residual), 1/Fs)

plt.figure()
plt.plot(fft_freq, np.abs(fft_vals))
plt.xlim(0, 10)
plt.title("Residual Spectrum After Learning (with envelope)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

# Extra: learned envelope vs true envelope (internal layer visualization)
e_learned = envelope(t, e0, Ce, De, env_freqs)
plt.figure()
plt.plot(t[:4000], env_true[:4000], label="True env")
plt.plot(t[:4000], e_learned[:4000], label="Learned env")
plt.title("Envelope Layer: learned vs true (zoom)")
plt.xlabel("Time (s)")
plt.ylabel("Envelope")
plt.legend()
plt.show()

# Extra: obstruction trace
plt.figure()
plt.plot(peak_hist)
plt.title("Dominant Residual Peak Frequency Over Time (obstruction trace)")
plt.xlabel("Training snapshot")
plt.ylabel("Peak frequency (Hz)")
plt.show()

print("Final carrier frequencies:", [float(f) for f in freqs])
print("Final carrier magnitudes:", mags(A, B))
print("Learned affine bias b:", float(b))
print("Learned envelope e0:", float(e0))
print("Learned envelope atoms (freqs):", env_freqs)
print("Ce:", [float(x) for x in Ce])
print("De:", [float(x) for x in De])
print("Final MSE:", float(E_hist[-1]))
print("Lift events (first 20):", lift_events[:20])
