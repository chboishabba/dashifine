import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# Periodic table: full table cloud that "lights up"
# -----------------------------
csv_path = "PeriodicTableCSV.csv"
df = pd.read_csv(csv_path)

Z = df["number"].astype(int).to_numpy()
sym = df["symbol"].astype(str).to_numpy()
name = df["name"].astype(str).to_numpy()

period = pd.to_numeric(df["period"], errors="coerce").to_numpy()
group  = pd.to_numeric(df["group"], errors="coerce").to_numpy()
en     = pd.to_numeric(df["electronegativity_pauling"], errors="coerce").to_numpy()
mass   = pd.to_numeric(df["atomic_mass"], errors="coerce").to_numpy()

# Handle NaNs: group should exist; EN sometimes missing -> 0
en = np.where(np.isfinite(en), en, 0.0)
mass = np.where(np.isfinite(mass), mass, np.nanmedian(mass))

def norm01(x):
    x = x.astype(float)
    m = np.nanmin(x); M = np.nanmax(x)
    y = (x - m) / (M - m + 1e-9)
    return np.clip(y, 0, 1)

F = np.stack([
    norm01(period),
    norm01(group),
    norm01(en),
    norm01(np.log(np.clip(mass, 1e-6, None)))
], axis=1)

# Simple deterministic 4D->3D linear projection (kept stable)
P = np.array([
    [ 1.0, -0.2,  0.3,  0.1],
    [-0.3,  1.0,  0.2, -0.1],
    [ 0.2,  0.1,  1.0,  0.3],
], dtype=np.float32)
X3 = (F @ P.T).astype(np.float32)
# Normalize to [-1,1] per axis
X3 = (X3 - X3.min(axis=0)) / (X3.max(axis=0) - X3.min(axis=0) + 1e-9)
X3 = (X3 - 0.5) * 2

nE = len(Z)
frames = nE  # one element per frame; loopable because it wraps
fps = 24

fig = plt.figure(figsize=(8.5, 7))
ax = fig.add_subplot(111, projection="3d")

def update(i):
    ax.cla()
    j = i % nE

    # Neighbor highlight: same group OR same period (plus self)
    same_group = (group == group[j])
    same_period = (period == period[j])
    neigh = same_group | same_period

    # Base scatter
    ax.scatter(X3[:,0], X3[:,1], X3[:,2], s=10, alpha=0.08)

    # Highlight neighbors
    ax.scatter(X3[neigh,0], X3[neigh,1], X3[neigh,2], s=18, alpha=0.35)

    # Current element bright
    ax.scatter([X3[j,0]], [X3[j,1]], [X3[j,2]], s=90, alpha=0.95)

    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    u = i/frames
    ax.view_init(20 + 6*np.sin(2*np.pi*u), 25 + 360*u)

    ax.set_title(f"Periodic Table Cloud (all 118) | highlight: group/period\n"
                 f"{int(Z[j])} {sym[j]} — {name[j]}  (period={int(period[j])}, group={int(group[j])})")
    return []

ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
pt_cloud_path = "periodic_table_cloud_lighting_full.gif"
ani.save(pt_cloud_path, writer=PillowWriter(fps=fps))
plt.close(fig)

# -----------------------------
# OFDM utilities (numpy only)
# -----------------------------
from scipy.ndimage import distance_transform_edt, label
from scipy.special import erf
from skimage.measure import marching_cubes, euler_number

def qpsk(rng, n):
    bits = rng.integers(0, 2, size=(n, 2))
    sym = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)
    sym /= np.sqrt(2)
    return sym.astype(np.complex64)

def qam16(rng, n):
    # Gray-like mapping via levels {-3,-1,1,3}
    levels = np.array([-3,-1,1,3], dtype=np.float32)
    idx = rng.integers(0, 4, size=(n,2))
    s = levels[idx[:,0]] + 1j*levels[idx[:,1]]
    s /= np.sqrt((np.mean(np.abs(s)**2))+1e-9)
    return s.astype(np.complex64)

def ofdm_symbol(rng, n_sc=64, cp=16, const="qpsk", pilots=False, pilot_every=4):
    X = np.zeros(n_sc, dtype=np.complex64)
    if const == "qpsk":
        data = qpsk(rng, n_sc)
    else:
        data = qam16(rng, n_sc)

    if pilots:
        pilot_idx = np.arange(0, n_sc, pilot_every)
        data_idx = np.setdiff1d(np.arange(n_sc), pilot_idx)
        X[data_idx] = data[data_idx]
        X[pilot_idx] = (1+1j)/np.sqrt(2)  # constant pilot
    else:
        X[:] = data

    x = np.fft.ifft(X).astype(np.complex64)
    x_cp = np.concatenate([x[-cp:], x]).astype(np.complex64)
    return X, x_cp

def multipath_channel(rng, L=5):
    taps = (rng.normal(size=L) + 1j*rng.normal(size=L)).astype(np.complex64)
    p = np.exp(-np.arange(L)/1.2).astype(np.float32)
    taps *= np.sqrt(p / (np.sum(p)+1e-9)).astype(np.float32)
    return taps

def apply_channel(x_cp, h):
    y = np.convolve(x_cp, h, mode="full")[:len(x_cp)].astype(np.complex64)
    return y

def add_awgn(rng, y, snr_db):
    sigp = np.mean(np.abs(y)**2).astype(np.float32)
    snr = 10**(snr_db/10)
    npow = sigp / (snr + 1e-9)
    n = (rng.normal(size=y.shape) + 1j*rng.normal(size=y.shape)).astype(np.complex64)
    n *= np.sqrt(npow/2).astype(np.float32)
    return (y + n).astype(np.complex64)

def inject_cfo_phase_noise(rng, y_cp, cfo_norm, phase_noise_std=0.0):
    # cfo_norm in subcarrier spacings: exp(j*2π*cfo*n/N)
    N = len(y_cp)
    n = np.arange(N, dtype=np.float32)
    rot = np.exp(1j*2*np.pi*cfo_norm*n/N).astype(np.complex64)
    y = (y_cp * rot).astype(np.complex64)
    if phase_noise_std > 0:
        # random-walk phase noise
        dphi = rng.normal(scale=phase_noise_std, size=N).astype(np.float32)
        phi = np.cumsum(dphi).astype(np.float32)
        y = (y * np.exp(1j*phi).astype(np.complex64)).astype(np.complex64)
    return y

def rx_fft(y_cp, n_sc=64, cp=16):
    y = y_cp[cp:cp+n_sc].astype(np.complex64)
    Y = np.fft.fft(y).astype(np.complex64)
    return Y

# Feature extractors (4D)
def feats_common(y_cp, n_sc=64, cp=16):
    a = y_cp[:cp]
    b = y_cp[n_sc:n_sc+cp]
    cp_corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)

    Y = rx_fft(y_cp, n_sc=n_sc, cp=cp)
    mag = np.abs(Y)
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))

    crest = float(np.max(np.abs(y_cp)) / (np.sqrt(np.mean(np.abs(y_cp)**2)) + 1e-9))
    return cp_corr, mag_mean, mag_std, crest, Y

def feats_channel_length(y_cp, n_sc=64, cp=16):
    cp_corr, mag_mean, mag_std, crest, Y = feats_common(y_cp, n_sc, cp)
    # Frequency response roughness proxy: variance of adjacent differences
    dmag = np.diff(np.abs(Y))
    rough = float(np.std(dmag))
    f = np.array([
        np.clip(cp_corr, 0, 1),
        np.clip(rough/2.0, 0, 1),
        np.clip(mag_std/2.0, 0, 1),
        np.clip((crest-1.0)/3.0, 0, 1),
    ], dtype=np.float32)
    return f

def feats_cfo(y_cp, n_sc=64, cp=16):
    cp_corr, mag_mean, mag_std, crest, Y = feats_common(y_cp, n_sc, cp)
    ph = np.angle(Y)
    dph = np.diff(ph)
    # wrap-safe dispersion
    ph_disp = float(np.mean(np.minimum(np.abs(dph), np.abs(dph - 2*np.pi))))
    # inter-carrier "tilt": mean phase slope
    slope = float(np.mean(np.unwrap(ph)[1:] - np.unwrap(ph)[:-1]))
    f = np.array([
        np.clip(cp_corr, 0, 1),
        np.clip(ph_disp/3.0, 0, 1),
        np.clip(np.abs(slope)/2.0, 0, 1),
        np.clip((crest-1.0)/3.0, 0, 1),
    ], dtype=np.float32)
    return f

def feats_evm(y_cp, X_true, h_true, n_sc=64, cp=16):
    # Oracle equalization to compute EVM (for labeling), but features are receiver-ish
    Y = rx_fft(y_cp, n_sc=n_sc, cp=cp)
    H = np.fft.fft(np.pad(h_true, (0, n_sc-len(h_true))), n=n_sc).astype(np.complex64)
    Xhat = Y / (H + 1e-9)
    evm = float(np.sqrt(np.mean(np.abs(Xhat - X_true)**2)) / (np.sqrt(np.mean(np.abs(X_true)**2)) + 1e-9))

    cp_corr, mag_mean, mag_std, crest, _ = feats_common(y_cp, n_sc, cp)
    f = np.array([
        np.clip(cp_corr, 0, 1),
        np.clip(mag_std/2.0, 0, 1),
        np.clip(mag_mean/3.0, 0, 1),
        np.clip((crest-1.0)/3.0, 0, 1),
    ], dtype=np.float32)
    return f, evm

def feats_pilots_ls(y_cp, n_sc=64, cp=16, pilot_every=4):
    Y = rx_fft(y_cp, n_sc=n_sc, cp=cp)
    pilot_idx = np.arange(0, n_sc, pilot_every)
    Xp = (1+1j)/np.sqrt(2)
    Hls = Y[pilot_idx] / (Xp + 1e-9)
    # roughness of estimated channel across pilots
    rough = float(np.std(np.diff(np.abs(Hls))))
    ph_disp = float(np.std(np.diff(np.unwrap(np.angle(Hls)))))
    crest = float(np.max(np.abs(y_cp)) / (np.sqrt(np.mean(np.abs(y_cp)**2)) + 1e-9))
    # also include cp_corr
    a = y_cp[:cp]; b = y_cp[n_sc:n_sc+cp]
    cp_corr = float(np.abs(np.vdot(a,b)) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
    f = np.array([
        np.clip(cp_corr, 0, 1),
        np.clip(rough/1.5, 0, 1),
        np.clip(ph_disp/2.0, 0, 1),
        np.clip((crest-1.0)/3.0, 0, 1),
    ], dtype=np.float32)
    return f

# Geometry functions (reuse)
def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def alpha_xyz(z0, w0, centers4, res=20):
    grid = np.linspace(-1, 1, res, dtype=np.float32)
    X, Y, Zg = np.meshgrid(grid, grid, grid, indexing="ij")
    Zs = Zg + z0
    S = np.zeros_like(X, dtype=np.float32)
    for c in centers4:
        dx = X - c[0]; dy = Y - c[1]; dz = Zs - c[2]; dw = w0 - c[3]
        d = np.sqrt(dx*dx + dy*dy + dz*dz + dw*dw, dtype=np.float32)
        S += gelu((1.0 - d) * 2.8).astype(np.float32)
    S = S + 1e-7
    return np.clip(S / S.max(), 0, 1).astype(np.float32)

def signed_distance(mask):
    return distance_transform_edt(mask) - distance_transform_edt(~mask)

def mesh_points(occ, res, max_pts=850, seed=0):
    rng = np.random.default_rng(seed)
    try:
        v, _, _, _ = marching_cubes(occ.astype(np.float32), level=0.5, spacing=(2/(res-1),)*3)
        vw = (v - (res-1)/2) * (2/(res-1))
        if len(vw) > max_pts:
            idx = rng.choice(len(vw), max_pts, replace=False)
            vw = vw[idx]
        return vw.astype(np.float32)
    except Exception:
        return np.zeros((0,3), dtype=np.float32)

def betti_fast(occ):
    occ = occ.astype(bool)
    _, b0 = label(occ)
    comp = ~occ
    _, ccomp = label(comp)
    b2 = max(0, ccomp - 1)
    chi = int(euler_number(occ, connectivity=1))
    b1 = b0 + b2 - chi
    if b1 < 0: b1 = 0
    return b0, b1, b2

def triangle_wave(u):
    return 1.0 - 4.0*np.abs(u - 0.5)  # [-1,1]

def sigmoid(x): return 1/(1+np.exp(-x))

# Base centers & mapping from normalized weights to drift
C0 = np.array([ 0.50,  0.40, -0.20,  0.30], dtype=np.float32)
M0 = np.array([-0.60,  0.10,  0.60, -0.40], dtype=np.float32)
Y0 = np.array([ 0.10, -0.50, -0.40,  0.50], dtype=np.float32)
K0 = np.array([-0.20, -0.30,  0.50, -0.60], dtype=np.float32)
B0 = np.stack([C0,M0,Y0,K0], axis=0)

Wcent = np.array([
    [ 0.30, -0.10,  0.15,  0.05],
    [-0.10,  0.30, -0.05,  0.15],
    [ 0.10,  0.05,  0.30, -0.10],
    [-0.15, -0.05,  0.10,  0.30],
], dtype=np.float32)

def render_task_gif(task_name, make_example_fn, feat_fn, label_fn, extra_text_fn, out_path):
    rng = np.random.default_rng(11)

    n_sc, cp = 64, 16
    res = 20
    base_level = 0.30
    radius = 0.35
    frames = 96
    fps = 24

    w = np.zeros(4, dtype=np.float32); b = 0.0; lr = 0.16

    meshes = []
    bettis = np.zeros((frames,3), dtype=int)
    volI = np.zeros(frames, dtype=np.float32)
    losses = np.zeros(frames, dtype=np.float32)
    probs = np.zeros(frames, dtype=np.float32)
    labs = np.zeros(frames, dtype=np.float32)
    feats = np.zeros((frames,4), dtype=np.float32)
    meta = []

    for i in range(frames):
        u = i/frames
        # periodic curriculum parameter for loopability
        cur = make_example_fn(rng, u)
        y_cp, feat_ctx = cur
        f = feat_fn(y_cp, feat_ctx)
        lab, lab_display = label_fn(y_cp, feat_ctx, u)
        labs[i] = lab
        feats[i] = f
        meta.append(lab_display)

        z = float(np.dot(w, f) + b)
        p = float(sigmoid(z))
        probs[i] = p
        loss = -(lab*np.log(p+1e-9) + (1-lab)*np.log(1-p+1e-9))
        losses[i] = loss

        g = (p - lab)
        w = (w - lr*g*f).astype(np.float32)
        b = float(b - lr*g)

        wn = w / (np.linalg.norm(w) + 1e-9)
        drift = (Wcent @ wn.astype(np.float32))
        centers = (B0 + 0.65*drift[None,:]).astype(np.float32)

        # orbit nudged by feature[0] and feature[3] (loopable)
        th = 2*np.pi*u
        z0 = float(np.cos(th)*radius + 0.18*(f[0]-0.5))
        w0 = float(np.sin(th)*radius + 0.18*(f[3]-0.5))

        A = alpha_xyz(z0,  w0, centers, res=res) >= base_level
        B = alpha_xyz(z0, -w0, centers, res=res) >= base_level
        blend = np.minimum(signed_distance(A), signed_distance(B))
        t = float(0.85 * triangle_wave(u))
        occ = blend > t

        meshes.append(mesh_points(occ, res=res, max_pts=850, seed=i))
        volI[i] = float(occ.mean())
        bettis[i] = betti_fast(occ)

    # rolling acc for plot
    acc = ((probs >= 0.5).astype(np.float32) == labs).astype(np.float32)
    win = 12
    roll_acc = np.convolve(acc, np.ones(win)/win, mode="same")

    fig = plt.figure(figsize=(11, 7.5))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_txt = fig.add_subplot(2, 2, 2); ax_txt.axis("off")
    ax1 = fig.add_subplot(2, 2, 3)
    ax2 = fig.add_subplot(2, 2, 4)

    x = np.arange(frames)

    def update(i):
        ax3d.cla(); ax_txt.cla(); ax_txt.axis("off")
        ax1.cla(); ax2.cla()

        pts = meshes[i]
        if len(pts):
            ax3d.scatter(pts[:,0], pts[:,1], pts[:,2], s=2, alpha=0.85)

        ax3d.set_xlim(-1,1); ax3d.set_ylim(-1,1); ax3d.set_zlim(-1,1)
        u = i/frames
        ax3d.view_init(18 + 6*np.sin(2*np.pi*u), 25 + 360*u)
        ax3d.set_title(f"{task_name}: 4D interpretation state → nested 3D geometry (loopable)")

        p = float(probs[i]); lab = int(labs[i]); pred = 1 if p >= 0.5 else 0
        f = feats[i]
        ax_txt.text(0.0, 0.86, f"Frame {i}/{frames-1}", fontsize=13)
        ax_txt.text(0.0, 0.74, meta[i], fontsize=15, weight="bold")
        ax_txt.text(0.0, 0.62, f"p(class1)={p:.2f}  pred={pred}  label={lab}", fontsize=13)
        ax_txt.text(0.0, 0.48, "Features (0..1):", fontsize=12)
        ax_txt.text(0.0, 0.36, f"f0={f[0]:.2f}  f1={f[1]:.2f}", fontsize=12)
        ax_txt.text(0.0, 0.25, f"f2={f[2]:.2f}  f3={f[3]:.2f}", fontsize=12)
        ax_txt.text(0.0, 0.10, f"Topology β={tuple(map(int,bettis[i]))}  volI={volI[i]:.3f}", fontsize=12)

        extra = extra_text_fn(i)
        if extra:
            ax_txt.text(0.0, 0.02, extra, fontsize=11)

        ax1.plot(x, losses, label="loss")
        ax1.axvline(i, linestyle="--")
        ax1.set_title("Online learning loss")
        ax1.legend(loc="upper right")

        ax2.plot(x, roll_acc, label=f"rolling acc (w={win})")
        ax2.plot(x, probs, label="p(class1)", alpha=0.7)
        ax2.axvline(i, linestyle="--")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title("Performance + belief")
        ax2.legend(loc="lower right")

        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000/24)
    ani.save(out_path, writer=PillowWriter(fps=24))
    plt.close(fig)

# -----------------------------
# Task 1: Channel length (L=3 vs L=7)
# -----------------------------
def make_ex_channel_len(rng, u):
    n_sc, cp = 64, 16
    # periodic switch of L for a loopable curriculum
    L = 3 if np.sin(2*np.pi*u) >= 0 else 7
    snr = 16.0 + 6.0*np.sin(2*np.pi*u + np.pi/3)
    X, x_cp = ofdm_symbol(rng, n_sc=n_sc, cp=cp, const="qpsk", pilots=False)
    h = multipath_channel(rng, L=L)
    y = add_awgn(rng, apply_channel(x_cp, h), snr)
    return y, {"L": L, "snr": snr}

def feat_channel_len(y_cp, ctx):
    return feats_channel_length(y_cp, n_sc=64, cp=16)

def label_channel_len(y_cp, ctx, u):
    L = ctx["L"]
    lab = 1.0 if L == 7 else 0.0
    return lab, f"Channel length class: L={L} taps  (class1=7 taps)  SNR≈{ctx['snr']:.1f} dB"

chlen_path = "ofdm_task_channel_length_L3_vs_L7.gif"
render_task_gif(
    "OFDM Channel Length",
    make_ex_channel_len,
    feat_channel_len,
    label_channel_len,
    lambda i: "",
    chlen_path
)

# -----------------------------
# Task 2: CFO + phase noise (low vs high CFO)
# -----------------------------
def make_ex_cfo(rng, u):
    n_sc, cp = 64, 16
    # CFO sweeps periodically; label based on magnitude
    cfo = 0.20*np.sin(2*np.pi*u)  # +/-0.2 subcarrier
    pn = 0.010 + 0.015*(0.5+0.5*np.sin(2*np.pi*u + np.pi/4))  # phase noise std
    snr = 18.0
    X, x_cp = ofdm_symbol(rng, n_sc=n_sc, cp=cp, const="qpsk", pilots=False)
    h = multipath_channel(rng, L=5)
    y = add_awgn(rng, apply_channel(x_cp, h), snr)
    y = inject_cfo_phase_noise(rng, y, cfo_norm=cfo, phase_noise_std=pn)
    return y, {"cfo": cfo, "pn": pn, "snr": snr}

def feat_cfo(y_cp, ctx):
    return feats_cfo(y_cp, n_sc=64, cp=16)

def label_cfo(y_cp, ctx, u):
    cfo = ctx["cfo"]
    lab = 1.0 if abs(cfo) >= 0.10 else 0.0
    return lab, f"CFO/PN class: CFO={cfo:+.2f} subcarrier  PNσ={ctx['pn']:.3f}  (class1=|CFO|≥0.10)"

cfo_path = "ofdm_task_cfo_phase_noise.gif"
render_task_gif(
    "OFDM CFO + Phase Noise",
    make_ex_cfo,
    feat_cfo,
    label_cfo,
    lambda i: "",
    cfo_path
)

# -----------------------------
# Task 3: EVM class (low vs high EVM)
# -----------------------------
def make_ex_evm(rng, u):
    n_sc, cp = 64, 16
    # SNR sweeps; so does channel length slightly
    snr = 6.0 + 20.0*(0.5+0.5*np.sin(2*np.pi*u))
    L = 3 if np.sin(2*np.pi*u + np.pi/6) >= 0 else 7
    X, x_cp = ofdm_symbol(rng, n_sc=n_sc, cp=cp, const="qpsk", pilots=False)
    h = multipath_channel(rng, L=L)
    y = add_awgn(rng, apply_channel(x_cp, h), snr)
    return y, {"X": X, "h": h, "snr": snr, "L": L}

def feat_evm(y_cp, ctx):
    f, evm = feats_evm(y_cp, ctx["X"], ctx["h"], n_sc=64, cp=16)
    ctx["evm"] = evm
    return f

def label_evm(y_cp, ctx, u):
    evm = ctx.get("evm", 0.0)
    lab = 1.0 if evm >= 0.35 else 0.0
    return lab, f"EVM class: EVM={evm:.2f}  SNR≈{ctx['snr']:.1f} dB  L={ctx['L']}  (class1=EVM≥0.35)"

evm_path = "ofdm_task_evm_class.gif"
render_task_gif(
    "OFDM EVM Prediction",
    make_ex_evm,
    feat_evm,
    label_evm,
    lambda i: "",
    evm_path
)

# -----------------------------
# Task 4: 16QAM + pilots, classify SNR bin (low vs high)
# -----------------------------
def make_ex_16qam_pilots(rng, u):
    n_sc, cp = 64, 16
    snr = 8.0 + 18.0*(0.5+0.5*np.sin(2*np.pi*u))
    X, x_cp = ofdm_symbol(rng, n_sc=n_sc, cp=cp, const="16qam", pilots=True, pilot_every=4)
    h = multipath_channel(rng, L=5)
    y = add_awgn(rng, apply_channel(x_cp, h), snr)
    return y, {"snr": snr}

def feat_16qam_pilots(y_cp, ctx):
    return feats_pilots_ls(y_cp, n_sc=64, cp=16, pilot_every=4)

def label_16qam_pilots(y_cp, ctx, u):
    snr = ctx["snr"]
    lab = 1.0 if snr >= 16.0 else 0.0
    return lab, f"16QAM+pilots SNR class: SNR≈{snr:.1f} dB  (class1=HIGH≥16 dB)"

qam_path = "ofdm_task_16qam_pilots_snr_class.gif"
render_task_gif(
    "OFDM 16QAM + Pilots",
    make_ex_16qam_pilots,
    feat_16qam_pilots,
    label_16qam_pilots,
    lambda i: "",
    qam_path
)

(pt_cloud_path, chlen_path, cfo_path, evm_path, qam_path)

