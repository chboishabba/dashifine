import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Adaptive oscillator learner (improved)
# 1) Linear A/B parameterization (sin+cos per freq)
# 2) Stabilized frequency gradients (use centered, scaled time in grad term)
# 3) Residual-driven structural lift (M10 analogue) with "symmetry-aware" handling:
#    - If peak is near existing freq: nudge/rotate that component instead of adding redundant duplicate
#    - Periodic merge of near-identical freqs (since duplicates are gauge-equivalent in A/B form)
# 4) Frequency repulsion regularizer (prevents accidental collapse; still allows closeness if needed)
# 5) Extra internals: component magnitudes, peak-tracking over time, residual spectrum
#
# Each plot is separate (no subplots) and we do not set explicit colors.
# =========================================================

# -----------------------------
# Target signal (same as your original)
# -----------------------------
T = 40.0
Fs = 800
t = np.linspace(0, T, int(T * Fs), endpoint=False)

true_freqs = np.array([1.0, 2.6, 4.2])
true_amps  = np.array([1.2, 0.8, 0.6])
true_phis  = np.array([0.3, 1.7, -0.8])

env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)

y_target = env * (
    true_amps[0] * np.sin(2*np.pi*true_freqs[0]*t + true_phis[0]) +
    true_amps[1] * np.sin(2*np.pi*true_freqs[1]*t + true_phis[1]) +
    true_amps[2] * np.sin(2*np.pi*true_freqs[2]*t + true_phis[2])
)

# -----------------------------
# Model init
# Start slightly under-specified to force lifts, but not too weak.
# -----------------------------
rng = np.random.default_rng(0)

freqs = [1.0, 2.0, 4.0]  # start missing 2.6 and 4.2 structure (4.0 is close-ish)
A = list(rng.normal(0.0, 0.15, size=len(freqs)))
B = list(rng.normal(0.0, 0.15, size=len(freqs)))

# Learning params
lr_A = 5e-3
lr_B = 5e-3
lr_f = 3e-6  # stabilized grad, can be larger than before
repel = 1e-4 # frequency repulsion strength

steps = 14000
batch = 1600

# Lift / obstruction params
lift_threshold = 0.12
max_oscillators = 7
delta_f = 0.18          # "near existing" frequency window (symmetry/gauge equivalence)
nudge_eta = 0.35        # if near existing, nudge it toward the peak instead of adding duplicate

# Merge params (gauge-fix duplicates)
merge_every = 400       # iterations
merge_delta = 0.12      # merge if freqs within this

# Logging
snap_every = 25
E_hist = []
freq_hist = []
mag_hist = []         # magnitudes sqrt(A^2+B^2)
peak_hist = []        # dominant residual peak frequency (for diagnostics)
lift_events = []      # (snapshot_index, new_freq, action)

def synth(A, B, freqs):
    y = np.zeros_like(t)
    for k in range(len(freqs)):
        y += A[k]*np.sin(2*np.pi*freqs[k]*t) + B[k]*np.cos(2*np.pi*freqs[k]*t)
    return y

def component_magnitudes(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.sqrt(A*A + B*B)

def merge_nearby(freqs, A, B, delta):
    """
    Merge components with nearly identical frequencies.
    In A/B form, two same-frequency components are gauge-equivalent to one
    with coefficients summed.
    """
    if len(freqs) <= 1:
        return freqs, A, B

    order = np.argsort(freqs)
    freqs_s = [float(freqs[i]) for i in order]
    A_s = [float(A[i]) for i in order]
    B_s = [float(B[i]) for i in order]

    out_f, out_A, out_B = [], [], []
    i = 0
    while i < len(freqs_s):
        f0 = freqs_s[i]
        a0 = A_s[i]
        b0 = B_s[i]
        j = i + 1
        # gather cluster
        while j < len(freqs_s) and abs(freqs_s[j] - f0) <= delta:
            # merge by summing coefficients
            a0 += A_s[j]
            b0 += B_s[j]
            # keep representative frequency as coefficient-weighted average
            w0 = abs(a0) + abs(b0) + 1e-12
            wj = abs(A_s[j]) + abs(B_s[j]) + 1e-12
            f0 = (w0*f0 + wj*freqs_s[j]) / (w0 + wj)
            j += 1
        out_f.append(float(f0))
        out_A.append(float(a0))
        out_B.append(float(b0))
        i = j

    return out_f, out_A, out_B

def dominant_residual_peak(residual, Fs, fmin=0.5, fmax=10.0):
    fft_vals = np.fft.rfft(residual)
    fft_freq = np.fft.rfftfreq(len(residual), 1/Fs)
    mask = (fft_freq >= fmin) & (fft_freq <= fmax)
    mags = np.abs(fft_vals[mask])
    if mags.size == 0:
        return None, None, None
    peak_i = int(np.argmax(mags))
    return float(fft_freq[mask][peak_i]), float(mags[peak_i]), (fft_freq, np.abs(fft_vals))

# -----------------------------
# Training loop
# -----------------------------
for it in range(steps):
    N = len(freqs)

    start = rng.integers(0, len(t) - batch)
    idx = slice(start, start + batch)
    tb = t[idx]
    yb = y_target[idx]

    # forward
    y_hat = np.zeros_like(yb)
    sin_mat = []
    cos_mat = []

    for k in range(N):
        arg = 2*np.pi*freqs[k]*tb
        s = np.sin(arg)
        c = np.cos(arg)
        sin_mat.append(s)
        cos_mat.append(c)
        y_hat += A[k]*s + B[k]*c

    sin_mat = np.array(sin_mat)
    cos_mat = np.array(cos_mat)

    err = y_hat - yb

    # grads A/B
    grad_A = 2*np.mean(err[None, :] * sin_mat, axis=1)
    grad_B = 2*np.mean(err[None, :] * cos_mat, axis=1)

    # stabilized time for frequency grad term (keep forward model exact)
    tau = tb - tb.mean()
    tau_scale = (tau.std() + 1e-9)
    tau = tau / tau_scale  # O(1)

    # grads freq + repulsion
    grad_f = np.zeros(N)
    for k in range(N):
        # d/d f of (A sin + B cos) = (A cos - B sin) * 2π t
        term = (A[k]*cos_mat[k] - B[k]*sin_mat[k]) * (2*np.pi * tau)
        grad_f[k] = 2*np.mean(err * term)

    # repulsion to reduce accidental collapse (soft; doesn't forbid closeness)
    for k in range(N):
        rep = 0.0
        for j in range(N):
            if j == k:
                continue
            rep += 1.0 / ((freqs[k] - freqs[j])**2 + 0.05)
        grad_f[k] += repel * rep

    # updates
    for k in range(N):
        A[k] -= lr_A * grad_A[k]
        B[k] -= lr_B * grad_B[k]
        freqs[k] -= lr_f * grad_f[k]

    # periodic merge (gauge-fix duplicates)
    if (it + 1) % merge_every == 0:
        freqs, A, B = merge_nearby(freqs, A, B, merge_delta)

    # snapshot + lift check
    if it % snap_every == 0:
        y_full = synth(A, B, freqs)
        mse = float(np.mean((y_full - y_target)**2))
        E_hist.append(mse)
        freq_hist.append(freqs.copy())
        mag_hist.append(component_magnitudes(A, B).tolist())

        # obstruction: look at dominant residual peak
        peak_f, peak_mag, _ = dominant_residual_peak(y_target - y_full, Fs, fmin=0.5, fmax=10.0)
        peak_hist.append(peak_f if peak_f is not None else np.nan)

        if (peak_f is not None) and (mse > lift_threshold) and (len(freqs) < max_oscillators):
            # if peak is near existing, nudge that component (symmetry/rotation handling)
            diffs = np.array([abs(peak_f - f) for f in freqs])
            k_near = int(np.argmin(diffs))
            if diffs[k_near] <= delta_f:
                freqs[k_near] = (1.0 - nudge_eta) * freqs[k_near] + nudge_eta * peak_f
                lift_events.append((len(E_hist)-1, peak_f, "NUDGE"))
            else:
                freqs.append(float(peak_f))
                A.append(0.0)
                B.append(0.0)
                lift_events.append((len(E_hist)-1, peak_f, "ADD"))

# Convert logs to arrays (ragged -> padded)
E_hist = np.array(E_hist)

max_len = max(len(f) for f in freq_hist)
freq_matrix = np.full((len(freq_hist), max_len), np.nan)
mag_matrix = np.full((len(mag_hist), max_len), np.nan)

for i, f in enumerate(freq_hist):
    freq_matrix[i, :len(f)] = f
    mag_matrix[i, :len(mag_hist[i])] = mag_hist[i]

y_learned = synth(A, B, freqs)

# -----------------------------
# PLOT 1: Target vs learned (same style as your original)
# -----------------------------
plt.figure()
win = slice(0, 2000)
plt.plot(t[win], y_target[win], label="Target")
plt.plot(t[win], y_learned[win], label="Learned")
plt.title("Adaptive Oscillator Fit (A/B + Lift + Frequency Dynamics)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# -----------------------------
# PLOT 2: Energy (MSE) over snapshots
# -----------------------------
plt.figure()
plt.plot(E_hist)
plt.title("Energy (MSE) During Training")
plt.xlabel("Training snapshot")
plt.ylabel("MSE")
plt.show()

# -----------------------------
# PLOT 3: Frequency trajectories (each oscillator line; new ones appear)
# -----------------------------
plt.figure()
for k in range(freq_matrix.shape[1]):
    plt.plot(freq_matrix[:, k])
plt.title("Frequency Trajectories (new lines indicate lifts)")
plt.xlabel("Training snapshot")
plt.ylabel("Frequency (Hz)")
plt.show()

# -----------------------------
# PLOT 4: Component magnitudes (sqrt(A^2+B^2)) as internal state strength
# -----------------------------
plt.figure()
for k in range(mag_matrix.shape[1]):
    plt.plot(mag_matrix[:, k])
plt.title("Component Magnitudes (internal strengths)")
plt.xlabel("Training snapshot")
plt.ylabel("Magnitude")
plt.show()

# -----------------------------
# PLOT 5: Residual spectrum (final)
# -----------------------------
residual = y_target - y_learned
fft_vals = np.fft.rfft(residual)
fft_freq = np.fft.rfftfreq(len(residual), 1/Fs)

plt.figure()
plt.plot(fft_freq, np.abs(fft_vals))
plt.xlim(0, 10)
plt.title("Residual Spectrum After Adaptive Learning")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

# Extra: peak tracking over time (internal "obstruction" diagnostic)
plt.figure()
plt.plot(peak_hist)
plt.title("Dominant Residual Peak Frequency Over Time (obstruction trace)")
plt.xlabel("Training snapshot")
plt.ylabel("Peak frequency (Hz)")
plt.show()

print("Final learned frequencies:", [float(f) for f in freqs])
print("Final component magnitudes:", component_magnitudes(A, B))
print("Final MSE:", float(E_hist[-1]))
print("Lift events (first 15):", lift_events[:15])
