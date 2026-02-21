# Generate requested outputs 1–6
import numpy as np, matplotlib.pyplot as plt, itertools, colorsys
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.special import erf
from scipy.ndimage import label, distance_transform_edt
from skimage.measure import marching_cubes, euler_number

# ---------- Field ----------
C_centers = [np.array([0.5, 0.4, -0.2, 0.3], dtype=np.float32)]
M_centers = [np.array([-0.6, 0.1, 0.6, -0.4], dtype=np.float32)]
Y_centers = [np.array([0.1, -0.5, -0.4, 0.5], dtype=np.float32)]
K_centers = [np.array([-0.2, -0.3, 0.5, -0.6], dtype=np.float32)]

def gelu(x): return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def alpha_xyz(z0, w0, res=24):
    grid = np.linspace(-1, 1, res, dtype=np.float32)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    Zs = Z + z0
    def one_class(centers):
        f = np.zeros_like(X, dtype=np.float32)
        for c in centers:
            dx = X - c[0]; dy = Y - c[1]; dz = Zs - c[2]; dw = w0 - c[3]
            d = np.sqrt(dx*dx + dy*dy + dz*dz + dw*dw, dtype=np.float32)
            f += gelu((1.0 - d) * 2.8).astype(np.float32)
        return f
    S = one_class(C_centers)+one_class(M_centers)+one_class(Y_centers)+one_class(K_centers)+1e-7
    return np.clip(S / S.max(), 0, 1).astype(np.float32)

def signed_distance(mask):
    return distance_transform_edt(mask) - distance_transform_edt(~mask)

def mesh_vertices(mask, res, max_pts=2000):
    try:
        v, _, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=(2/(res-1),)*3)
        vw = (v - (res-1)/2) * (2/(res-1))
        if len(vw) > max_pts:
            idx = np.random.choice(len(vw), max_pts, replace=False)
            vw = vw[idx]
        return vw.astype(np.float32)
    except:
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

# ---------- Orbit ----------
frames = 16
radius = 0.35
res = 24
level = 0.28
thetas = np.linspace(0, 2*np.pi, frames, endpoint=False)

meshes = []
sigma = []
dm = []
bettis = []

for th in thetas:
    z0 = float(np.cos(th)*radius); w0 = float(np.sin(th)*radius)
    A = alpha_xyz(z0,w0,res=res)>=level
    B = alpha_xyz(z0,-w0,res=res)>=level
    occ = np.minimum(signed_distance(A), signed_distance(B))>0
    meshes.append(mesh_vertices(occ,res=res,max_pts=1800))
    volA=A.mean(); volB=B.mean(); volI=occ.mean()
    s3=1 if volA>=0.66 else (-1 if volA<=0.33 else 0)
    s9=1 if volB>=0.66 else (-1 if volB<=0.33 else 0)
    s6=1 if volI>=0.66 else (-1 if volI<=0.33 else 0)
    sigma.append(int(s3!=s6)+int(s6!=s9))
    dm.append(int(s3!=s9))
    bettis.append(betti_fast(occ))

sigma=np.array(sigma); dm=np.array(dm); bettis=np.array(bettis)

# ---------- 1 Δm coloring ----------
def make_gif(values, title, path, cmap):
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111,projection="3d")
    vmin,vmax=float(values.min()),float(values.max())
    if vmax==vmin: vmax=vmin+1
    def update(i):
        ax.cla()
        pts=meshes[i]
        if len(pts):
            t=(values[i]-vmin)/(vmax-vmin)
            ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=[cmap(t)],s=3)
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.set_title(f"{title}={int(values[i])}")
        ax.view_init(18,20+i*10)
        return []
    ani=FuncAnimation(fig,update,frames=frames,interval=150)
    ani.save(path,writer=PillowWriter(fps=6))
    plt.close(fig)

gif1="invariant_Delta_m_bounds_-1_1.gif"
make_gif(dm,"Δm",gif1,plt.cm.plasma)

# ---------- 2 (σ,Δm) HSV ----------
fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111,projection="3d")
sig_max=max(1,int(sigma.max())); dm_max=max(1,int(dm.max()))
def hsv(sig,dmv):
    h=(dmv/dm_max) if dm_max else 0
    v=0.4+0.6*(sig/sig_max) if sig_max else 0.5
    return colorsys.hsv_to_rgb(h,0.9,v)
def update_sd(i):
    ax.cla()
    pts=meshes[i]
    if len(pts):
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],
                   c=[hsv(int(sigma[i]),int(dm[i]))],s=3)
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_title(f"(σ,Δm)=({int(sigma[i])},{int(dm[i])})")
    ax.view_init(18,20+i*10)
    return []
ani=FuncAnimation(fig,update_sd,frames=frames,interval=150)
gif2="invariant_sigma_Delta_m_hsv_bounds_-1_1.gif"
ani.save(gif2,writer=PillowWriter(fps=6))
plt.close(fig)

# ---------- 3 Betti over θ ----------
fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)
def update_b(i):
    ax.cla()
    ax.plot(range(i+1),bettis[:i+1,0])
    ax.plot(range(i+1),bettis[:i+1,1])
    ax.plot(range(i+1),bettis[:i+1,2])
    ax.set_ylim(0,max(1,int(bettis.max()+1)))
    ax.set_title("β0, β1, β2 over θ")
    return []
ani=FuncAnimation(fig,update_b,frames=frames,interval=150)
gif3="betti012_over_theta.gif"
ani.save(gif3,writer=PillowWriter(fps=6))
plt.close(fig)

# ---------- 4 Filtration curves ----------
thresholds=np.linspace(-1.5,1.5,15)
fil=np.zeros((frames,len(thresholds)))
for i,th in enumerate(thetas):
    z0=float(np.cos(th)*radius); w0=float(np.sin(th)*radius)
    A=alpha_xyz(z0,w0,res=res)>=level
    B=alpha_xyz(z0,-w0,res=res)>=level
    blend=np.minimum(signed_distance(A),signed_distance(B))
    for j,t in enumerate(thresholds):
        occ=blend>t
        fil[i,j]=label(occ)[1]

fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)
def update_f(i):
    ax.cla()
    ax.plot(thresholds,fil[i])
    ax.set_title(f"Filtration β0 curve θ={int(np.degrees(thetas[i]))}")
    return []
ani=FuncAnimation(fig,update_f,frames=frames,interval=150)
gif4="persistence_surrogate_filtration.gif"
ani.save(gif4,writer=PillowWriter(fps=6))
plt.close(fig)

# ---------- 5 p-adic embedding ----------
states=np.array(list(itertools.product([-1,0,1],repeat=9)),dtype=np.int32)
digits=(states+1).astype(np.int64)
powers=(3**np.arange(9)).astype(np.int64)
N=(digits*powers).sum(axis=1)
primes=np.array([71,59,47])
pad=np.stack([(N%p)/(p-1) for p in primes],axis=1)
pad=(pad-0.5)*2

fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111,projection="3d")
def update_pad(i):
    ax.cla()
    ax.scatter(pad[:,0],pad[:,1],pad[:,2],s=2)
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_title("3^9 p-adic residue embedding")
    ax.view_init(20,20+i*10)
    return []
ani=FuncAnimation(fig,update_pad,frames=frames,interval=150)
gif5="ternary_kernel_padic_residue.gif"
ani.save(gif5,writer=PillowWriter(fps=6))
plt.close(fig)

# ---------- 6 Box-count dimension ----------
all_pts=np.vstack([p for p in meshes if len(p)]) if any(len(p) for p in meshes) else np.zeros((0,3))
eps=[1/2**k for k in range(2,7)]
counts=[]
for e in eps:
    if len(all_pts)==0: counts.append(0)
    else:
        P=(all_pts+1)/2
        bins=np.floor(P/e).astype(int)
        counts.append(len({tuple(x) for x in bins}))
x=np.log(1/np.array(eps)); y=np.log(np.array(counts)+1e-9)
slope=np.polyfit(x,y,1)[0] if len(counts)>1 else np.nan

fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(111)
ax.plot(x,y,marker="o")
ax.set_title(f"Box-count dimension ≈ {slope:.3f}")
ax.set_xlabel("log(1/ε)"); ax.set_ylabel("log N(ε)")
png6="boxcount_dimension_invariant_attractor.png"
plt.tight_layout(); plt.savefig(png6,dpi=200); plt.close(fig)

(gif1,gif2,gif3,gif4,gif5,png6)
