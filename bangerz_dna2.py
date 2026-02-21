# Cumulative manifold animation + repeated motif count over time
# Using real Vidania genome and Haar band-energy pipeline
# Two separate GIFs (each single plot, no specific colors)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# -------- Load FASTA --------
fasta_path = "VFCOTACU.fasta"

def load_fasta(path):
    seq = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.append(line.strip().upper())
    return "".join(seq)

seq = load_fasta(fasta_path)

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

# -------- PCA embedding --------
pca = PCA(n_components=3)
coords = pca.fit_transform(E)
coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)

# -------- Similarity --------
norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
E_norm = E / norms
similarity = E_norm @ E_norm.T

# Threshold for motif repetition
threshold = 0.95

# Precompute cumulative motif counts
motif_counts = []
cumulative_edges = []
for i in range(W):
    sims = similarity[i, :i]
    repeats = np.where(sims > threshold)[0]
    motif_counts.append(len(repeats))
    cumulative_edges.append(repeats)

motif_counts = np.array(motif_counts)

# -------- Animation 1: cumulative manifold --------
frames = min(W, 120)
fps = 20

fig1 = plt.figure(figsize=(7,7))
ax1 = fig1.add_subplot(111, projection="3d")

def update1(i):
    ax1.cla()
    ax1.set_title(f"Cumulative Haar manifold (window {i+1})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")

    # plot all previous points
    ax1.scatter(coords[:i+1,0], coords[:i+1,1], coords[:i+1,2], s=8)

    # draw repetition edges
    for j in cumulative_edges[i]:
        ax1.plot([coords[i,0], coords[j,0]],
                 [coords[i,1], coords[j,1]],
                 [coords[i,2], coords[j,2]])

    return []

ani1 = FuncAnimation(fig1, update1, frames=frames, interval=1000/fps)
out1 = "vidania_cumulative_manifold.gif"
ani1.save(out1, writer=PillowWriter(fps=fps))
plt.close(fig1)

# -------- Animation 2: repeated motif count over time --------
fig2 = plt.figure(figsize=(7,5))
ax2 = fig2.add_subplot(111)

def update2(i):
    ax2.cla()
    ax2.set_title("Repeated motif count over genome")
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("# repeated motifs (similarity > 0.95)")
    ax2.plot(np.arange(i+1), motif_counts[:i+1])
    return []

ani2 = FuncAnimation(fig2, update2, frames=frames, interval=1000/fps)
out2 = "vidania_repeated_motif_count2.gif"
ani2.save(out2, writer=PillowWriter(fps=fps))
plt.close(fig2)

out1, out2
