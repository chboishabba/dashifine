import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Define two base oscillatory modes (standing waves)
wave1 = np.sin(2 * X + 1.5 * Y)
wave2 = np.sin(1.2 * X - 2.0 * Y + 1)

# Superposition (interference pattern)
Psi = wave1 + wave2

# Gradient descent surface = interference energy landscape
E = (wave1 - wave2)**2

fig = plt.figure(figsize=(10, 6))

# Plot 1: Interference pattern
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Psi, cmap='viridis', linewidth=0, alpha=0.9)
ax1.set_title('Interference Pattern (Memory Wave Field)')
ax1.set_xlabel('Axis 1 (Sensory)')
ax1.set_ylabel('Axis 2 (Affective)')
ax1.set_zlabel('Amplitude')

# Plot 2: Energy surface (Gradient Descent Landscape)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, E, cmap='inferno', linewidth=0, alpha=0.9)
ax2.set_title('Gradient Descent Energy Surface')
ax2.set_xlabel('Phase shift X')
ax2.set_ylabel('Phase shift Y')
ax2.set_zlabel('Interference Energy')

plt.tight_layout()
plt.show()


# ---
# Simulation: Coupled oscillators learning a target waveform by minimizing interference energy
# - A small set of oscillators (amplitude a_k, phase phi_k, fixed frequency f_k)
# - Learn a_k and phi_k via gradient descent to approximate a target signal y(t)
# - Optional Kuramoto-style coupling on phases to encourage coordination
# - Track energy (MSE), show convergence, and visualize final fit

import numpy as np
import matplotlib.pyplot as plt

# 1) Target signal (can be any waveform)
T = 4.0                    # seconds
Fs = 800                   # samples per second
t = np.linspace(0, T, int(T*Fs), endpoint=False)

# Compose a slightly complex target signal (non-harmonic mix + envelope)
true_freqs = np.array([1.0, 2.6, 4.2])   # Hz
true_amps  = np.array([1.2, 0.8, 0.6])
true_phis  = np.array([0.3, 1.7, -0.8])

env = 0.5 + 0.5*np.sin(2*np.pi*0.25*t)   # slow envelope

y_target = env * (
    true_amps[0]*np.sin(2*np.pi*true_freqs[0]*t + true_phis[0]) +
    true_amps[1]*np.sin(2*np.pi*true_freqs[1]*t + true_phis[1]) +
    true_amps[2]*np.sin(2*np.pi*true_freqs[2]*t + true_phis[2])
)

# 2) Model: N oscillators with fixed frequencies (could be learned too, but keep simple/stable)
N = 4
freqs = np.array([1.0, 2.0, 3.0, 4.0])       # Hz (basis set)

rng = np.random.default_rng(42)
a = rng.normal(0.0, 0.2, size=N)            # amplitudes
phi = rng.normal(0.0, 0.5, size=N)           # phases (radians)

# 3) Learning hyperparameters
lr_a   = 5e-3     # learning rate for amplitudes
lr_phi = 5e-3     # learning rate for phases
kappa  = 1e-2     # Kuramoto coupling strength between phases (coordination)
lam_a  = 1e-4     # L2 on amplitudes (regularization)
lam_p  = 0.0      # L2 on phases (optional)

steps = 1200      # gradient steps
batch = 800       # samples per step (mini-batch size)

# For logging
E_hist = []
a_hist = []
phi_hist = []

# Helper to synthesize current model output
def synth(a, phi):
    S = np.zeros_like(t)
    for k in range(N):
        S += a[k]*np.sin(2*np.pi*freqs[k]*t + phi[k])
    return S

# Stochastic gradient descent loop
for it in range(steps):
    # sample a contiguous window (mini-batch) to speed/regularize learning
    start = rng.integers(0, len(t)-batch)
    idx = slice(start, start+batch)
    tb = t[idx]
    yb = y_target[idx]

    # forward on batch
    y_hat = np.zeros_like(yb)
    sin_mat = np.zeros((N, batch))
    cos_mat = np.zeros((N, batch))
    for k in range(N):
        arg = 2*np.pi*freqs[k]*tb + phi[k]
        s = np.sin(arg)
        c = np.cos(arg)
        sin_mat[k] = s
        cos_mat[k] = c
        y_hat += a[k]*s

    err = y_hat - yb
    E = np.mean(err**2)

    # Gradients (d/d a_k) and (d/d phi_k)
    # dE/da_k = 2 * mean(err * sin)
    # dE/dphi_k = 2 * mean(err * a_k * cos)
    grad_a = 2.0*np.mean(err[None, :] * sin_mat, axis=1) + 2*lam_a*a
    grad_phi = 2.0*np.mean((err[None, :] * (a[:, None]*cos_mat)), axis=1) + 2*lam_p*phi

    # Kuramoto-style phase coupling to encourage phase alignment (coordination)
    # d/dphi_k += -kappa * sum_j sin(phi_j - phi_k)
    for k in range(N):
        coupling = 0.0
        for j in range(N):
            if j == k: 
                continue
            coupling += np.sin(phi[j] - phi[k])
        grad_phi[k] += -kappa * coupling

    # Parameter updates
    a  -= lr_a   * grad_a
    phi-= lr_phi * grad_phi

    # Log full-trajectory diagnostics occasionally
    if it % 5 == 0 or it == steps-1:
        y_full = synth(a, phi)
        E_hist.append(np.mean((y_full - y_target)**2))
        a_hist.append(a.copy())
        phi_hist.append(phi.copy())

# Convert logs to arrays
E_hist  = np.array(E_hist)
a_hist  = np.stack(a_hist, axis=0)
phi_hist= np.stack(phi_hist, axis=0)

# Final synthesis
y_learned = synth(a, phi)

# 4) Plots: target vs learned; energy curve; parameter trajectories
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# (A) Target vs learned on a short window for clarity
win = slice(0, 2000)
axes[0].plot(t[win], y_target[win], label='Target', linewidth=1.5)
axes[0].plot(t[win], y_learned[win], label='Learned', linestyle='--', linewidth=1.0)
axes[0].set_title('Coupled Oscillators Learning a Waveform')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].legend()

# (B) Energy (MSE) over training snapshots
axes[1].plot(np.arange(len(E_hist)), E_hist)
axes[1].set_title('Interference Energy (MSE) during Learning')
axes[1].set_xlabel('Training snapshot')
axes[1].set_ylabel('MSE')

# (C) Parameter trajectories (amplitudes and phases)
steps_logged = np.arange(len(a_hist))
for k in range(N):
    axes[2].plot(steps_logged, a_hist[:, k], linewidth=1.0, label=f'a{k+1}')
# Add a twin axis for phases to avoid scale conflict
ax2 = axes[2].twinx()
for k in range(N):
    ax2.plot(steps_logged, phi_hist[:, k], linestyle='--', linewidth=1.0)
axes[2].set_title('Parameter Trajectories (solid: amplitudes, dashed: phases)')
axes[2].set_xlabel('Training snapshot')
axes[2].set_ylabel('Amplitude')
ax2.set_ylabel('Phase (rad)')

plt.tight_layout()
plt.show()
