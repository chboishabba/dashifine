# Real Vidania genome (FASTA) + Haar band-energy + interconnected repetitions
# Single 3D plot (no subplots), no explicit colors, no orbit.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# -------- Load FASTA --------
fasta_path = "/home/c/Downloads/VFCOTACU.fasta"

def load_fasta(path):
    seq = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.append(line.strip().upper())
    return "".join(seq)

seq = load_fasta(fasta_path)
genome_len = len(seq)

# -------- Haar band-energy pipeline --------
BASES = "ACGT"
BASE_TO_IDX = {b:i for i,b in enumerate(BASES)}

def one_hot(s):
    X = np.zeros((4, len(s)), dtype=np.float32)
    for i, b in enumerate(s):
        j = BASE_TO_IDX.get(b, None)
        if j is not None:
            X[j, i] = 1.0
    return X

def haar_detail(signal, levels):
    arr = signal.astype(np.float32)
    details = []
    for _ in range(levels):
        if len(arr) < 2:
            details.append(np.zeros(1, dtype=np.float32))
            break
        even = (len(arr)//2)*2
        arr = arr[:even]
        avg = (arr[0::2] + arr[1::2]) / 2.0
        diff = (arr[0::2] - arr[1::2]) / 2.0
        details.append(diff)
        arr = avg
    while len(details) < levels:
        details.append(np.zeros(1, dtype=np.float32))
    return details

def band_energies(sequence, window=512, step=512, levels=5):
    num = 1 + (len(sequence)-window)//step
    E = np.zeros((num, 4*levels), dtype=np.float32)
    for t in range(num):
        s = sequence[t*step : t*step+window]
        X = one_hot(s)
        k = 0
        for ch in range(4):
            details = haar_detail(X[ch], levels)
            for d in details:
                E[t, k] = float(np.mean(np.abs(d)))
                k += 1
    norms = E.mean(axis=0, keepdims=True) + 1e-8
    return E / norms

window = 512
step = 512
levels = 5

E = band_energies(seq, window, step, levels)
W = E.shape[0]

# -------- PCA embedding to 3D --------
pca = PCA(n_components=3)
coords = pca.fit_transform(E)
coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)

# -------- Precompute similarities (cosine) for repetition links --------
norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
E_norm = E / norms
similarity = E_norm @ E_norm.T  # cosine similarity matrix

# -------- Animation (single 3D plot, no orbit) --------
frames = min(120, W)
fps = 20

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")

def update(i):
    ax.cla()
    ax.set_title(f"Vidania 50kb — Haar band-energy manifold (window {i+1}/{W})")
    ax.set_xlim(coords[:,0].min(), coords[:,0].max())
    ax.set_ylim(coords[:,1].min(), coords[:,1].max())
    ax.set_zlim(coords[:,2].min(), coords[:,2].max())
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # Plot all points (manifold)
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=5, alpha=0.2)

    # Highlight current window
    ax.scatter(coords[i,0], coords[i,1], coords[i,2], s=60)

    # Draw links to top 3 similar past windows (repetition structure)
    sims = similarity[i].copy()
    sims[i] = -1
    idx = np.argsort(sims)[-3:]
    for j in idx:
        ax.plot([coords[i,0], coords[j,0]],
                [coords[i,1], coords[j,1]],
                [coords[i,2], coords[j,2]])

    return []

ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
out_path = "vidania_real_haar_repetition_no_orbit.gif"
ani.save(out_path, writer=PillowWriter(fps=fps))
plt.close(fig)

out_path
