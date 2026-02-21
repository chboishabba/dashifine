import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# -----------------------------
# "Clean demonstration" (toy, but principled):
# - Work in melodies x ∈ (Z_12)^n
# - Define a symmetry operator S (reverse + inversion)
# - Define asymmetry energy E_sym(x) = sum circular_distance(x_i, S(x)_i)
# - Define a contractive flow x <- round((1-α)x + α S(x))  (then mod 12)
# - Define projection P onto a small codebook of prototypes (some symmetric, some generic)
#   using energy d(x, y) + β * E_sym(y).  (β biases toward symmetric attractors)
# - Canonicalize C by transposition to 0 (pitch-offset quotient)
# - Basin volume = fraction of random initial states mapping to each attractor under T = C∘P∘(flow)
# -----------------------------

rng = np.random.default_rng(7)

N_PITCH = 12

def mod12(x):
    return np.mod(x, N_PITCH)

def circ_dist(a, b):
    """Circular distance on Z12 (min steps). Vectorized."""
    d = np.abs(a - b) % N_PITCH
    return np.minimum(d, N_PITCH - d)

def symmetry_S(m):
    """Reverse + inversion (I ∘ retrograde), about 0."""
    return mod12(-m[::-1])

def E_sym(m):
    s = symmetry_S(m)
    return float(np.sum(circ_dist(m, s)))

def flow_step(m, alpha=0.35):
    """One contraction step toward S(m) in Z12 using rounding in R then mod 12."""
    s = symmetry_S(m).astype(float)
    m_f = m.astype(float)
    m_new = (1 - alpha) * m_f + alpha * s
    return mod12(np.rint(m_new).astype(int))

def canon_transpose_to_zero(m):
    """Quotient out global transposition: shift so first note becomes 0."""
    k = int(m[0]) % N_PITCH
    return mod12(m - k)

def d_melody(x, y):
    """Distance between melodies (sum of circular distances)."""
    return float(np.sum(circ_dist(x, y)))

@dataclass(frozen=True)
class Prototype:
    name: str
    melody: np.ndarray  # canonical (transposed to 0)
    kind: str          # "symmetric" or "generic"

def make_symmetric_prototype(n):
    """Create a melody with S(m)=m by choosing first half and mirroring."""
    # Choose free variables for first ceil(n/2)
    half = (n + 1) // 2
    a = rng.integers(0, N_PITCH, size=half)
    # Build full melody satisfying m[i] = -m[n-1-i]
    m = np.zeros(n, dtype=int)
    for i in range(n):
        j = n - 1 - i
        if i < half:
            m[i] = int(a[i])
        else:
            m[i] = int((-m[j]) % N_PITCH)
    return canon_transpose_to_zero(mod12(m))

def make_generic_prototype(n):
    m = rng.integers(0, N_PITCH, size=n)
    return canon_transpose_to_zero(mod12(m))

def snap_to_prototype(x, prototypes, beta=0.0):
    """Projection P: choose y minimizing d(x,y) + beta*E_sym(y)."""
    best = None
    best_val = 1e18
    for p in prototypes:
        val = d_melody(x, p.melody) + beta * E_sym(p.melody)
        if val < best_val:
            best_val = val
            best = p
    return best

def T_operator(x, prototypes, beta=0.0, alpha=0.35, steps=6):
    """T = C∘P∘flow^steps"""
    z = x.copy()
    for _ in range(steps):
        z = flow_step(z, alpha=alpha)
    z = canon_transpose_to_zero(z)
    p = snap_to_prototype(z, prototypes, beta=beta)
    return p, p.melody.copy()

# -----------------------------
# Build a small codebook
# -----------------------------
n = 8
protos = []
for i in range(3):
    protos.append(Prototype(f"Sym-{i+1}", make_symmetric_prototype(n), "symmetric"))
for i in range(3):
    protos.append(Prototype(f"Gen-{i+1}", make_generic_prototype(n), "generic"))

# Show prototype stats
proto_table = [(p.name, p.kind, E_sym(p.melody), p.melody.tolist()) for p in protos]
proto_table[:3], proto_table[3:]

# -----------------------------
# Basin volume vs beta
# -----------------------------
def estimate_basins(beta, M=6000):
    counts = {p.name: 0 for p in protos}
    for _ in range(M):
        x0 = rng.integers(0, N_PITCH, size=n)
        p, _ = T_operator(x0, protos, beta=beta)
        counts[p.name] += 1
    # Aggregate by kind
    sym = sum(counts[p.name] for p in protos if p.kind == "symmetric") / M
    gen = sum(counts[p.name] for p in protos if p.kind == "generic") / M
    return counts, sym, gen

betas = np.linspace(0, 12, 13)  # increasing preference for symmetry
sym_fracs = []
gen_fracs = []
by_proto = {p.name: [] for p in protos}

for b in betas:
    counts, sym, gen = estimate_basins(b, M=6000)
    sym_fracs.append(sym)
    gen_fracs.append(gen)
    for p in protos:
        by_proto[p.name].append(counts[p.name] / 6000)

# Plot: basin share by kind
plt.figure(figsize=(8, 4.5))
plt.plot(betas, sym_fracs, marker="o", label="Symmetric prototypes (total basin share)")
plt.plot(betas, gen_fracs, marker="o", label="Generic prototypes (total basin share)")
plt.xlabel("β (symmetry bias in projection energy)")
plt.ylabel("Estimated basin volume (fraction of random initial melodies)")
plt.title("Basin volumes under T = C ∘ P ∘ flow, as symmetry bias β increases")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Plot: basin share per prototype (to show redistribution)
plt.figure(figsize=(9, 5))
for p in protos:
    plt.plot(betas, by_proto[p.name], marker="o", label=p.name)
plt.xlabel("β (symmetry bias)")
plt.ylabel("Basin volume")
plt.title("Basin volumes for individual attractors (canonical prototypes)")
plt.grid(True, alpha=0.3)
plt.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show()

# -----------------------------
# Demonstrate contraction flow of E_sym over iterations for random starts
# -----------------------------
def flow_trajectory(x0, alpha=0.35, T=20):
    xs = [x0.copy()]
    es = [E_sym(x0)]
    x = x0.copy()
    for _ in range(T):
        x = flow_step(x, alpha=alpha)
        xs.append(x.copy())
        es.append(E_sym(x))
    return np.array(xs), np.array(es)

plt.figure(figsize=(8, 4.5))
for i in range(12):
    x0 = rng.integers(0, N_PITCH, size=n)
    _, es = flow_trajectory(x0, alpha=0.35, T=20)
    plt.plot(range(len(es)), es, alpha=0.7)
plt.xlabel("Iteration t")
plt.ylabel("Asymmetry energy E_sym(x_t)")
plt.title("Contractive flow toward symmetry: E_sym decreases over time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Animation: show one melody converging and then snapping to an attractor
# -----------------------------
x0 = rng.integers(0, N_PITCH, size=n)
xs, es = flow_trajectory(x0, alpha=0.35, T=18)
# pick a beta where symmetry dominates
beta_demo = 10.0
p_final, y_final = T_operator(x0, protos, beta=beta_demo, steps=18)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.set_ylim(-0.5, 11.5)
ax.set_xlim(-0.5, n - 0.5)
ax.set_xlabel("Time index i")
ax.set_ylabel("Pitch class (0..11)")
ax.set_title(f"One trajectory under flow; then projection to attractor (β={beta_demo:g})")
ax.grid(True, alpha=0.25)

line, = ax.plot([], [], marker="o")
text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

def init():
    line.set_data([], [])
    text.set_text("")
    return line, text

def update(frame):
    if frame < len(xs):
        m = xs[frame]
        line.set_data(np.arange(n), m)
        text.set_text(f"E_sym={E_sym(m):.1f}")
    else:
        # final snap display
        line.set_data(np.arange(n), y_final)
        text.set_text(f"SNAP → {p_final.name} ({p_final.kind}), E_sym={E_sym(y_final):.1f}")
    return line, text

anim = FuncAnimation(fig, update, frames=len(xs)+8, init_func=init, interval=300, blit=True)
out_gif = "/mnt/data/dashi_musical_symmetry_clean_demo.gif"
anim.save(out_gif, writer="pillow", fps=3)
plt.close(fig)

out_gif
