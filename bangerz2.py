import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.special import erf
from scipy.ndimage import distance_transform_edt

# ---------- helpers ----------
def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def alpha_xyz(z0, w0, centers4, res=16):
    g = np.linspace(-1, 1, res, dtype=np.float32)
    X, Y, Zg = np.meshgrid(g, g, g, indexing="ij")
    Zs = Zg + z0
    S = np.zeros_like(X, dtype=np.float32)
    for c in centers4:
        dx = X - c[0]; dy = Y - c[1]; dz = Zs - c[2]; dw = w0 - c[3]
        d = np.sqrt(dx*dx + dy*dy + dz*dz + dw*dw, dtype=np.float32)
        S += gelu((1.0 - d) * 2.6).astype(np.float32)
    S = S + 1e-7
    return np.clip(S / S.max(), 0, 1).astype(np.float32)

def signed_distance(mask):
    return distance_transform_edt(mask) - distance_transform_edt(~mask)

def triangle_wave(u):
    return 1.0 - 4.0*np.abs(u - 0.5)  # [-1,1]

def surface_points_from_occ(occ, max_pts=650, seed=0):
    occ = occ.astype(bool)
    b = occ & (
        (~np.roll(occ,  1, 0)) | (~np.roll(occ, -1, 0)) |
        (~np.roll(occ,  1, 1)) | (~np.roll(occ, -1, 1)) |
        (~np.roll(occ,  1, 2)) | (~np.roll(occ, -1, 2))
    )
    idx = np.argwhere(b)
    if idx.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(idx) > max_pts:
        idx = idx[rng.choice(len(idx), size=max_pts, replace=False)]
    return idx.astype(np.float32)

def idx_to_xyz(idx, res):
    return (idx/(res-1))*2 - 1

# ---------- periodic table cloud lighting (full 118) ----------
df = pd.read_csv("/mnt/data/PeriodicTableCSV.csv")
Z = df["number"].astype(int).to_numpy()
sym = df["symbol"].astype(str).to_numpy()
name = df["name"].astype(str).to_numpy()
period = pd.to_numeric(df["period"], errors="coerce").to_numpy()
group  = pd.to_numeric(df["group"], errors="coerce").to_numpy()
en     = pd.to_numeric(df["electronegativity_pauling"], errors="coerce").to_numpy()
mass   = pd.to_numeric(df["atomic_mass"], errors="coerce").to_numpy()
en = np.where(np.isfinite(en), en, 0.0)
mass = np.where(np.isfinite(mass), mass, np.nanmedian(mass))

def norm01(x):
    x = x.astype(float)
    m, M = np.nanmin(x), np.nanmax(x)
    return np.clip((x-m)/(M-m+1e-9), 0, 1)

F = np.stack([norm01(period), norm01(group), norm01(en), norm01(np.log(np.clip(mass, 1e-6, None)))], axis=1)
P = np.array([[ 1.0, -0.2,  0.3,  0.1],
              [-0.3,  1.0,  0.2, -0.1],
              [ 0.2,  0.1,  1.0,  0.3]], dtype=np.float32)
X3 = (F @ P.T).astype(np.float32)
X3 = (X3 - X3.min(axis=0)) / (X3.max(axis=0) - X3.min(axis=0) + 1e-9)
X3 = (X3 - 0.5) * 2

nE = len(Z)
fig = plt.figure(figsize=(8.0, 6.5))
ax = fig.add_subplot(111, projection="3d")

def pt_update(i):
    ax.cla()
    j = i % nE
    neigh = (group == group[j]) | (period == period[j])

    ax.scatter(X3[:,0], X3[:,1], X3[:,2], s=8, alpha=0.05)
    ax.scatter(X3[neigh,0], X3[neigh,1], X3[neigh,2], s=14, alpha=0.35)
    ax.scatter([X3[j,0]], [X3[j,1]], [X3[j,2]], s=110, alpha=0.95)

    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    u = i/nE
    ax.view_init(20 + 6*np.sin(2*np.pi*u), 25 + 360*u)
    ax.set_title(f"Periodic Table Cloud (118) — lit neighborhood\n{int(Z[j])} {sym[j]} — {name[j]}")
    return []

pt_ani = FuncAnimation(fig, pt_update, frames=nE, interval=1000/24)
pt_cloud_path = "/mnt/data/periodic_table_cloud_lighting_full.gif"
pt_ani.save(pt_cloud_path, writer=PillowWriter(fps=24))
plt.close(fig)

# ---------- single OFDM multi-task GIF (4 tasks in one file) ----------
def sigmoid(x): return 1/(1+np.exp(-x))

def qpsk(rng, n):
    bits = rng.integers(0, 2, size=(n, 2))
    s = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)
    return (s/np.sqrt(2)).astype(np.complex64)

def qam16(rng, n):
    levels = np.array([-3,-1,1,3], dtype=np.float32)
    idx = rng.integers(0, 4, size=(n,2))
    s = levels[idx[:,0]] + 1j*levels[idx[:,1]]
    s = s / np.sqrt(np.mean(np.abs(s)**2) + 1e-9)
    return s.astype(np.complex64)

def ofdm_symbol(rng, n_sc=64, cp=16, const="qpsk", pilots=False, pilot_every=4):
    X = np.zeros(n_sc, dtype=np.complex64)
    data = qpsk(rng, n_sc) if const=="qpsk" else qam16(rng, n_sc)
    if pilots:
        pilot_idx = np.arange(0, n_sc, pilot_every)
        data_idx = np.setdiff1d(np.arange(n_sc), pilot_idx)
        X[data_idx] = data[data_idx]
        X[pilot_idx] = (1+1j)/np.sqrt(2)
    else:
        X[:] = data
    x = np.fft.ifft(X).astype(np.complex64)
    return X, np.concatenate([x[-cp:], x]).astype(np.complex64)

def multipath_channel(rng, L=5):
    taps = (rng.normal(size=L) + 1j*rng.normal(size=L)).astype(np.complex64)
    p = np.exp(-np.arange(L)/1.2).astype(np.float32)
    taps *= np.sqrt(p/(np.sum(p)+1e-9)).astype(np.float32)
    return taps

def apply_channel(x_cp, h):
    return np.convolve(x_cp, h, mode="full")[:len(x_cp)].astype(np.complex64)

def add_awgn(rng, y, snr_db):
    sigp = np.mean(np.abs(y)**2).astype(np.float32)
    snr = 10**(snr_db/10)
    npow = sigp/(snr+1e-9)
    n = (rng.normal(size=y.shape) + 1j*rng.normal(size=y.shape)).astype(np.complex64)
    n *= np.sqrt(npow/2).astype(np.float32)
    return (y+n).astype(np.complex64)

def inject_cfo_phase_noise(rng, y_cp, cfo_norm, phase_noise_std=0.0):
    N = len(y_cp)
    n = np.arange(N, dtype=np.float32)
    rot = np.exp(1j*2*np.pi*cfo_norm*n/N).astype(np.complex64)
    y = (y_cp*rot).astype(np.complex64)
    if phase_noise_std>0:
        dphi = rng.normal(scale=phase_noise_std, size=N).astype(np.float32)
        phi = np.cumsum(dphi).astype(np.float32)
        y = (y*np.exp(1j*phi).astype(np.complex64)).astype(np.complex64)
    return y

def rx_fft(y_cp, n_sc=64, cp=16):
    return np.fft.fft(y_cp[cp:cp+n_sc]).astype(np.complex64)

def feats_common(y_cp, n_sc=64, cp=16):
    a = y_cp[:cp]; b = y_cp[n_sc:n_sc+cp]
    cp_corr = float(np.abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    Y = rx_fft(y_cp, n_sc, cp)
    mag = np.abs(Y)
    mag_mean = float(np.mean(mag)); mag_std = float(np.std(mag))
    crest = float(np.max(np.abs(y_cp))/(np.sqrt(np.mean(np.abs(y_cp)**2))+1e-9))
    return cp_corr, mag_mean, mag_std, crest, Y

def feats_channel_length(y_cp):
    cp_corr, mag_mean, mag_std, crest, Y = feats_common(y_cp)
    rough = float(np.std(np.diff(np.abs(Y))))
    return np.array([np.clip(cp_corr,0,1), np.clip(rough/2.0,0,1), np.clip(mag_std/2.0,0,1), np.clip((crest-1)/3.0,0,1)], dtype=np.float32)

def feats_cfo(y_cp):
    cp_corr, mag_mean, mag_std, crest, Y = feats_common(y_cp)
    ph = np.unwrap(np.angle(Y))
    dph = np.diff(ph)
    ph_disp = float(np.std(dph))
    slope = float(np.mean(dph))
    return np.array([np.clip(cp_corr,0,1), np.clip(ph_disp/2.0,0,1), np.clip(np.abs(slope)/1.5,0,1), np.clip((crest-1)/3.0,0,1)], dtype=np.float32)

def feats_evm(y_cp):
    cp_corr, mag_mean, mag_std, crest, Y = feats_common(y_cp)
    return np.array([np.clip(cp_corr,0,1), np.clip(mag_std/2.0,0,1), np.clip(mag_mean/3.0,0,1), np.clip((crest-1)/3.0,0,1)], dtype=np.float32)

def feats_pilots_ls(y_cp, pilot_every=4):
    Y = rx_fft(y_cp)
    pilot_idx = np.arange(0, 64, pilot_every)
    Xp = (1+1j)/np.sqrt(2)
    Hls = Y[pilot_idx]/(Xp+1e-9)
    rough = float(np.std(np.diff(np.abs(Hls))))
    ph_disp = float(np.std(np.diff(np.unwrap(np.angle(Hls)))))
    a = y_cp[:16]; b = y_cp[64:64+16]
    cp_corr = float(np.abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    crest = float(np.max(np.abs(y_cp))/(np.sqrt(np.mean(np.abs(y_cp)**2))+1e-9))
    return np.array([np.clip(cp_corr,0,1), np.clip(rough/1.5,0,1), np.clip(ph_disp/2.0,0,1), np.clip((crest-1)/3.0,0,1)], dtype=np.float32)

# 4 tasks in segments
tasks = [
    ("Channel length (L=3 vs L=7)", "class1=L7"),
    ("CFO/phase noise (|CFO|≥0.10)", "class1=high CFO"),
    ("EVM class (EVM≥0.35)", "class1=high EVM"),
    ("16QAM+pilots SNR (≥16dB)", "class1=high SNR"),
]

# geometry base centers
C0 = np.array([ 0.50,  0.40, -0.20,  0.30], dtype=np.float32)
M0 = np.array([-0.60,  0.10,  0.60, -0.40], dtype=np.float32)
Y0 = np.array([ 0.10, -0.50, -0.40,  0.50], dtype=np.float32)
K0 = np.array([-0.20, -0.30,  0.50, -0.60], dtype=np.float32)
B0 = np.stack([C0,M0,Y0,K0], axis=0)
Wcent = np.array([[ 0.30, -0.10,  0.15,  0.05],
                  [-0.10,  0.30, -0.05,  0.15],
                  [ 0.10,  0.05,  0.30, -0.10],
                  [-0.15, -0.05,  0.10,  0.30]], dtype=np.float32)

rng = np.random.default_rng(23)
frames = 80
seg = frames // 4
res = 16
base_level = 0.32
radius = 0.35

w = np.zeros(4, dtype=np.float32); b = 0.0; lr = 0.20

pts_list = []
titles = []
infos = []
probs = []
labs = []
feat_list = []

for i in range(frames):
    u = i/frames
    t = i // seg
    t = min(t, 3)
    local_u = (i - t*seg) / seg  # 0..1 within task
    # periodic curriculum
    if t == 0:
        L = 3 if np.sin(2*np.pi*local_u) >= 0 else 7
        snr = 14.0 + 6.0*np.sin(2*np.pi*local_u + np.pi/3)
        X, x_cp = ofdm_symbol(rng, const="qpsk")
        h = multipath_channel(rng, L=L)
        y = add_awgn(rng, apply_channel(x_cp, h), snr)
        f = feats_channel_length(y)
        lab = 1.0 if L==7 else 0.0
        info = f"L={L}  SNR≈{snr:.1f} dB"
    elif t == 1:
        cfo = 0.22*np.sin(2*np.pi*local_u)
        pn = 0.010 + 0.015*(0.5+0.5*np.sin(2*np.pi*local_u + np.pi/4))
        snr = 18.0
        X, x_cp = ofdm_symbol(rng, const="qpsk")
        h = multipath_channel(rng, L=5)
        y = add_awgn(rng, apply_channel(x_cp, h), snr)
        y = inject_cfo_phase_noise(rng, y, cfo_norm=cfo, phase_noise_std=pn)
        f = feats_cfo(y)
        lab = 1.0 if abs(cfo)>=0.10 else 0.0
        info = f"CFO={cfo:+.2f}  PNσ={pn:.3f}"
    elif t == 2:
        snr = 6.0 + 20.0*(0.5+0.5*np.sin(2*np.pi*local_u))
        L = 3 if np.sin(2*np.pi*local_u + np.pi/6) >= 0 else 7
        X, x_cp = ofdm_symbol(rng, const="qpsk")
        h = multipath_channel(rng, L=L)
        y = add_awgn(rng, apply_channel(x_cp, h), snr)
        # oracle EVM for label
        Y = rx_fft(y)
        H = np.fft.fft(np.pad(h, (0, 64-len(h))), n=64).astype(np.complex64)
        Xhat = Y/(H+1e-9)
        evm = float(np.sqrt(np.mean(np.abs(Xhat - X)**2)) / (np.sqrt(np.mean(np.abs(X)**2))+1e-9))
        f = feats_evm(y)
        lab = 1.0 if evm>=0.35 else 0.0
        info = f"EVM={evm:.2f}  SNR≈{snr:.1f} dB"
    else:
        snr = 8.0 + 18.0*(0.5+0.5*np.sin(2*np.pi*local_u))
        X, x_cp = ofdm_symbol(rng, const="16qam", pilots=True, pilot_every=4)
        h = multipath_channel(rng, L=5)
        y = add_awgn(rng, apply_channel(x_cp, h), snr)
        f = feats_pilots_ls(y, pilot_every=4)
        lab = 1.0 if snr>=16.0 else 0.0
        info = f"SNR≈{snr:.1f} dB"

    # online update
    p = float(sigmoid(float(np.dot(w,f)+b)))
    g = (p - lab)
    w = (w - lr*g*f).astype(np.float32)
    b = float(b - lr*g)

    wn = w/(np.linalg.norm(w)+1e-9)
    drift = (Wcent @ wn.astype(np.float32))
    centers = (B0 + 0.65*drift[None,:]).astype(np.float32)

    th = 2*np.pi*u
    z0 = float(np.cos(th)*radius + 0.18*(f[0]-0.5))
    w0 = float(np.sin(th)*radius + 0.18*(f[3]-0.5))

    A = alpha_xyz(z0,  w0, centers, res=res) >= base_level
    B = alpha_xyz(z0, -w0, centers, res=res) >= base_level
    blend = np.minimum(signed_distance(A), signed_distance(B))
    tthr = float(0.85*triangle_wave(u))
    occ = blend > tthr

    idx = surface_points_from_occ(occ, max_pts=650, seed=i)
    pts = idx_to_xyz(idx, res)

    pts_list.append(pts)
    titles.append(tasks[t][0])
    infos.append(info)
    probs.append(p)
    labs.append(lab)
    feat_list.append(f)

probs = np.array(probs, dtype=np.float32)
labs = np.array(labs, dtype=np.float32)
feat_list = np.array(feat_list, dtype=np.float32)

fig = plt.figure(figsize=(9.5, 6.5))
ax3d = fig.add_subplot(1,2,1, projection="3d")
ax_txt = fig.add_subplot(1,2,2); ax_txt.axis("off")

def ofdm_update(i):
    ax3d.cla(); ax_txt.cla(); ax_txt.axis("off")
    pts = pts_list[i]
    if len(pts):
        ax3d.scatter(pts[:,0], pts[:,1], pts[:,2], s=4, alpha=0.85)
    ax3d.set_xlim(-1,1); ax3d.set_ylim(-1,1); ax3d.set_zlim(-1,1)
    u = i/frames
    ax3d.view_init(18 + 6*np.sin(2*np.pi*u), 25 + 360*u)
    ax3d.set_title("OFDM multi-task: interpretation state → nested 3D")

    p = float(probs[i]); lab = int(labs[i]); pred = 1 if p>=0.5 else 0
    f = feat_list[i]
    ax_txt.text(0.0, 0.86, titles[i], fontsize=14, weight="bold")
    ax_txt.text(0.0, 0.74, infos[i], fontsize=12)
    ax_txt.text(0.0, 0.60, f"p(class1)={p:.2f}  pred={pred}  label={lab}", fontsize=12)
    ax_txt.text(0.0, 0.42, f"features: {f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f}, {f[3]:.2f}", fontsize=12)
    return []

ofdm_ani = FuncAnimation(fig, ofdm_update, frames=frames, interval=1000/24)
ofdm_multi_path = "/mnt/data/ofdm_multitask_interpretation_loop.gif"
ofdm_ani.save(ofdm_multi_path, writer=PillowWriter(fps=24))
plt.close(fig)

(pt_cloud_path, ofdm_multi_path)

