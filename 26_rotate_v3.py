import numpy as np, matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.special import erf
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
C_centers = [np.array([0.5, 0.4, -0.2, 0.3], dtype=np.float32)]
M_centers = [np.array([-0.6, 0.1, 0.6, -0.4], dtype=np.float32)]
Y_centers = [np.array([0.1, -0.5, -0.4, 0.5], dtype=np.float32)]
K_centers = [np.array([-0.2, -0.3, 0.5, -0.6], dtype=np.float32)]

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def class_field_points(P, centers, sharpness=2.8):
    Xp,Yp,Zp,Wp = P[:,0],P[:,1],P[:,2],P[:,3]
    f = np.zeros(len(P), dtype=np.float32)
    for c in centers:
        dx = Xp - c[0]; dy = Yp - c[1]; dz = Zp - c[2]; dw = Wp - c[3]
        d = np.sqrt(dx*dx+dy*dy+dz*dz+dw*dw, dtype=np.float32)
        f += gelu((1.0-d)*sharpness).astype(np.float32)
    return f

def cmyk_rgbS_at_points(P, sharpness=2.8, tie_gamma=0.9, penalty_strength=0.35):
    C = class_field_points(P, C_centers, sharpness)
    M = class_field_points(P, M_centers, sharpness)
    Yf= class_field_points(P, Y_centers, sharpness)
    Kk= class_field_points(P, K_centers, sharpness)
    stack = np.stack([C,M,Yf,Kk], axis=0)
    max1 = np.max(stack, axis=0)
    m = stack == max1[None,:]
    stack_masked = np.where(m, -np.inf, stack)
    max2 = np.max(stack_masked, axis=0)
    tie_pen = gelu(-tie_gamma*(max1-max2).astype(np.float32)).astype(np.float32)
    for f in (C,M,Yf,Kk):
        f *= (1 - penalty_strength * tie_pen)
    S = C+M+Yf+Kk + 1e-7
    wC,wM,wY,wK = C/S, M/S, Yf/S, Kk/S
    R = np.clip((1 - wM) * (1 - wK), 0, 1)
    G = np.clip((1 - wY) * (1 - wK), 0, 1)
    B = np.clip((1 - wC) * (1 - wK), 0, 1)
    rgb = np.stack([R,G,B], axis=1).astype(np.float32)
    return rgb, S

def alpha_xyz_wfixed(w0, res=64, sharpness=2.8, tie_gamma=0.9, penalty_strength=0.35):
    grid = np.linspace(-1,1,res, dtype=np.float32)
    X,Y,Z = np.meshgrid(grid, grid, grid, indexing="ij")
    def one_class(centers):
        f = np.zeros_like(X, dtype=np.float32)
        for c in centers:
            dx = X - c[0]; dy = Y - c[1]; dz = Z - c[2]; dw = w0 - c[3]
            d = np.sqrt(dx*dx+dy*dy+dz*dz+dw*dw, dtype=np.float32)
            f += gelu((1.0-d)*sharpness).astype(np.float32)
        return f
    C = one_class(C_centers); M = one_class(M_centers); Yf = one_class(Y_centers); Kk = one_class(K_centers)
    stack = np.stack([C,M,Yf,Kk], axis=0)
    max1 = np.max(stack, axis=0)
    m = stack == max1[None,...]
    stack_masked = np.where(m, -np.inf, stack)
    max2 = np.max(stack_masked, axis=0)
    tie_pen = gelu(-tie_gamma*(max1-max2).astype(np.float32)).astype(np.float32)
    for f in (C,M,Yf,Kk):
        f *= (1 - penalty_strength * tie_pen)
    S = C+M+Yf+Kk + 1e-7
    alpha = np.clip(S / S.max(), 0, 1).astype(np.float32)
    return alpha

# ---- Static outputs (fast) ----
# A/C: 3D isosurface of 4D field with w fixed
res = 70
alpha = alpha_xyz_wfixed(0.0, res=res)
level = 0.25
v,f,_,_ = marching_cubes(alpha, level=level, spacing=(2/(res-1),)*3)
vw = (v - (res-1)/2) * (2/(res-1))

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(vw[:,0], vw[:,1], vw[:,2], triangles=f, linewidth=0.08, alpha=0.9)
ax.set_title(f"A/C) 3D isosurface of 4D CMYK field (fix w=0), level={level}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=35)
iso_w0_png = "/mnt/data/cmyk_A_isosurface_fixw.png"
plt.savefig(iso_w0_png, dpi=200, bbox_inches="tight")
plt.close(fig)

# D: 3D projection of 4D object by integrating over w
def projected_alpha(res=64, w_samples=7):
    grid = np.linspace(-1,1,res, dtype=np.float32)
    X,Y,Z = np.meshgrid(grid, grid, grid, indexing="ij")
    ws = np.linspace(-1,1,w_samples, dtype=np.float32)
    acc = np.zeros_like(X, dtype=np.float32)
    for w0 in ws:
        a = alpha_xyz_wfixed(float(w0), res=res)  # already normalized per w0
        acc += a
    acc = acc / (acc.max() + 1e-9)
    return acc

resP = 62
proj = projected_alpha(res=resP, w_samples=7)
levelP = 0.35
vP,fP,_,_ = marching_cubes(proj, level=levelP, spacing=(2/(resP-1),)*3)
vPw = (vP - (resP-1)/2) * (2/(resP-1))

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(vPw[:,0], vPw[:,1], vPw[:,2], triangles=fP, linewidth=0.08, alpha=0.9)
ax.set_title(f"B/D) 3D projection of 4D object (integrate w), level={levelP}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=35)
proj_png = "/mnt/data/cmyk_D_projected_isosurface.png"
plt.savefig(proj_png, dpi=200, bbox_inches="tight")
plt.close(fig)

# B: 4D -> 3D embeddings (PCA + Isomap)
rng = np.random.default_rng(4)
P = rng.uniform(-1,1, size=(100000,4)).astype(np.float32)
rgbP, SP = cmyk_rgbS_at_points(P)
keep = 18000
idx = np.argpartition(SP, -keep)[-keep:]
P_keep, rgb_keep = P[idx], rgbP[idx]

pca = PCA(n_components=3, random_state=0)
Ypca = pca.fit_transform(P_keep)
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Ypca[:,0], Ypca[:,1], Ypca[:,2], c=rgb_keep, s=2, alpha=0.35)
ax.set_title("B) Fix nothing: 4D high-density points → 3D (PCA)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.view_init(elev=20, azim=35)
pca_png = "/mnt/data/cmyk_B_4d_to_3d_pca.png"
plt.savefig(pca_png, dpi=200, bbox_inches="tight")
plt.close(fig)

sub = 3500
sub_idx = rng.choice(len(P_keep), size=sub, replace=False)
P_sub, rgb_sub = P_keep[sub_idx], rgb_keep[sub_idx]
iso = Isomap(n_neighbors=16, n_components=3)
Yiso = iso.fit_transform(P_sub)
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Yiso[:,0], Yiso[:,1], Yiso[:,2], c=rgb_sub, s=3, alpha=0.45)
ax.set_title("B) Fix nothing: 4D manifold → 3D (Isomap)")
ax.set_xlabel("Iso1"); ax.set_ylabel("Iso2"); ax.set_zlabel("Iso3")
ax.view_init(elev=20, azim=35)
isomap_png = "/mnt/data/cmyk_B_4d_to_3d_isomap.png"
plt.savefig(isomap_png, dpi=200, bbox_inches="tight")
plt.close(fig)

# A: volumetric cloud (fix w=0) using random sampling in xyz for speed
# sample xyz points and color by class weights at (x,y,z,w0)
w0 = 0.0
M = 35000
xyz = rng.uniform(-1,1, size=(M,3)).astype(np.float32)
wcol = np.full((M,1), w0, dtype=np.float32)
Pxyz = np.concatenate([xyz, wcol], axis=1)
rgb_xyz, S_xyz = cmyk_rgbS_at_points(Pxyz)
# keep stronger points
th = np.quantile(S_xyz, 0.85)
m = S_xyz >= th
xyz2, rgb2 = xyz[m], rgb_xyz[m]
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz2[:,0], xyz2[:,1], xyz2[:,2], c=rgb2, s=2, alpha=0.35)
ax.set_title("A) Fix w=0: 3D volumetric cloud (top 15% density)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=35)
cloud_png = "/mnt/data/cmyk_A_cloud_fixw.png"
plt.savefig(cloud_png, dpi=200, bbox_inches="tight")
plt.close(fig)

# E: orbit animation (fast): scatter isosurface vertices (no trisurf) across (z0,w0) circle
frames = 18
resA = 44
levelA = 0.28
thetas = np.linspace(0, 2*np.pi, frames, endpoint=False)

def alpha_xyz_shift(z0, w0, res=resA):
    grid = np.linspace(-1,1,res, dtype=np.float32)
    X,Y,Z = np.meshgrid(grid, grid, grid, indexing="ij")
    Zs = Z + z0
    # compute alpha quickly by reusing class_field formula in 3D
    def one_class(centers):
        f = np.zeros_like(X, dtype=np.float32)
        for c in centers:
            dx = X - c[0]; dy = Y - c[1]; dz = Zs - c[2]; dw = w0 - c[3]
            d = np.sqrt(dx*dx+dy*dy+dz*dz+dw*dw, dtype=np.float32)
            f += gelu((1.0-d)*2.8).astype(np.float32)
        return f
    C = one_class(C_centers); M = one_class(M_centers); Yf = one_class(Y_centers); Kk = one_class(K_centers)
    S = C+M+Yf+Kk + 1e-7
    return np.clip(S / S.max(), 0, 1).astype(np.float32)

mesh_pts = []
for th in thetas:
    z0 = float(np.cos(th)*0.35)
    w0 = float(np.sin(th)*0.35)
    a = alpha_xyz_shift(z0, w0, res=resA)
    try:
        vv, ff, _, _ = marching_cubes(a, level=levelA, spacing=(2/(resA-1),)*3)
        vvw = (vv - (resA-1)/2) * (2/(resA-1))
        # downsample vertices for speed
        if len(vvw) > 6000:
            pick = rng.choice(len(vvw), size=6000, replace=False)
            vvw = vvw[pick]
        mesh_pts.append(vvw)
    except Exception:
        mesh_pts.append(np.zeros((0,3), dtype=np.float32))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([],[],[], s=4, alpha=0.6)

def upd(i):
    ax.cla()
    pts = mesh_pts[i]
    if len(pts):
        # color by z just to show depth
        c = (pts[:,2] - pts[:,2].min()) / (pts[:,2].ptp() + 1e-9)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=c, s=4, alpha=0.6)
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=20, azim=35)
    ax.set_title(f"E) Orbit (θ={np.rad2deg(thetas[i]):.0f}°): isosurface vertices")
    return []

ani = FuncAnimation(fig, upd, frames=frames, interval=160)
orbit_gif = "/mnt/data/cmyk_E_orbit_isosurface_points.gif"
ani.save(orbit_gif, writer=PillowWriter(fps=8))
plt.close(fig)

iso_w0_png, cloud_png, pca_png, isomap_png, proj_png, orbit_gif
