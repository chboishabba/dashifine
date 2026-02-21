import numpy as np, matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.special import erf

# Parameters (from demo_rgba)
C_centers = [np.array([0.5, 0.4, -0.2, 0.3], dtype=np.float32)]
M_centers = [np.array([-0.6, 0.1, 0.6, -0.4], dtype=np.float32)]
Y_centers = [np.array([0.1, -0.5, -0.4, 0.5], dtype=np.float32)]
K_centers = [np.array([-0.2, -0.3, 0.5, -0.6], dtype=np.float32)]

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def class_field_xyz(X,Y,Z,w0,centers,sharpness=2.8):
    f = np.zeros_like(X, dtype=np.float32)
    for c in centers:
        dx = X - c[0]
        dy = Y - c[1]
        dz = Z - c[2]
        dw = w0 - c[3]
        d = np.sqrt(dx*dx + dy*dy + dz*dz + dw*dw, dtype=np.float32)
        f += gelu((1.0 - d) * sharpness).astype(np.float32)
    return f

def cmyk_fields_xyz(w0, res=72, sharpness=2.8, tie_gamma=0.9, penalty_strength=0.35):
    grid = np.linspace(-1,1,res, dtype=np.float32)
    X,Y,Z = np.meshgrid(grid, grid, grid, indexing="ij")
    C = class_field_xyz(X,Y,Z,w0, C_centers, sharpness)
    M = class_field_xyz(X,Y,Z,w0, M_centers, sharpness)
    Yf= class_field_xyz(X,Y,Z,w0, Y_centers, sharpness)
    K = class_field_xyz(X,Y,Z,w0, K_centers, sharpness)
    stack = np.stack([C,M,Yf,K], axis=0)
    max1 = np.max(stack, axis=0)
    mask = stack == max1[None,...]
    stack_masked = np.where(mask, -np.inf, stack)
    max2 = np.max(stack_masked, axis=0)
    tie_pen = gelu(-tie_gamma*(max1-max2).astype(np.float32)).astype(np.float32)
    for f in (C,M,Yf,K):
        f *= (1 - penalty_strength * tie_pen)
    S = C+M+Yf+K + 1e-7
    wC,wM,wY,wK = C/S, M/S, Yf/S, K/S
    R = (1 - wM) * (1 - wK)
    G = (1 - wY) * (1 - wK)
    B = (1 - wC) * (1 - wK)
    alpha = np.clip(S / S.max(), 0, 1).astype(np.float32)
    # clip rgb safely
    R = np.clip(R,0,1); G=np.clip(G,0,1); B=np.clip(B,0,1)
    return (X,Y,Z), (R,G,B,alpha), (C,M,Yf,K,S)

# A) Fix w -> 3D isosurface + volumetric cloud
res = 72
(X,Y,Z), (R,G,B,alpha), _ = cmyk_fields_xyz(w0=0.0, res=res)
vol = alpha
level = 0.25
verts, faces, normals, values = marching_cubes(vol, level=level, spacing=(2/(res-1),)*3)
verts_world = (verts - (res-1)/2) * (2/(res-1))

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts_world[:,0], verts_world[:,1], verts_world[:,2], triangles=faces, linewidth=0.1, alpha=0.9)
ax.set_title(f"3D isosurface of 4D CMYK field (fix w=0), level={level}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=35)
iso_path = "cmyk_3d_isosurface_w0.png"
plt.savefig(iso_path, dpi=200, bbox_inches="tight")
plt.close(fig)

mask = alpha > 0.18
coords = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
rgb = np.stack([R[mask], G[mask], B[mask]], axis=1)
rgb = np.clip(rgb, 0, 1)
if len(coords) > 30000:
    idx = np.random.default_rng(0).choice(len(coords), size=30000, replace=False)
    coords = coords[idx]; rgb = rgb[idx]
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=rgb, s=2, alpha=0.35)
ax.set_title("3D volumetric cloud (fix w=0), colored by CMYK→RGB")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=35)
cloud_path = "cmyk_3d_cloud_w0.png"
plt.savefig(cloud_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# B) Fix nothing -> 4D points -> 3D embedding (PCA + Isomap)
def fields_at_points(P, sharpness=2.8, tie_gamma=0.9, penalty_strength=0.35):
    Xp,Yp,Zp,Wp = P[:,0],P[:,1],P[:,2],P[:,3]
    def one_class(centers):
        f = np.zeros(len(P), dtype=np.float32)
        for c in centers:
            dx = Xp - c[0]; dy = Yp - c[1]; dz = Zp - c[2]; dw = Wp - c[3]
            d = np.sqrt(dx*dx+dy*dy+dz*dz+dw*dw, dtype=np.float32)
            f += gelu((1.0-d)*sharpness).astype(np.float32)
        return f
    C = one_class(C_centers); M = one_class(M_centers); Yf = one_class(Y_centers); Kk = one_class(K_centers)
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

rng = np.random.default_rng(2)
P = rng.uniform(-1,1, size=(120000,4)).astype(np.float32)
rgbP, SP = fields_at_points(P)
k_keep = 20000
idx = np.argpartition(SP, -k_keep)[-k_keep:]
P_keep = P[idx]; rgb_keep = rgbP[idx]

pca = PCA(n_components=3, random_state=0)
Ypca = pca.fit_transform(P_keep)
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Ypca[:,0], Ypca[:,1], Ypca[:,2], c=rgb_keep, s=2, alpha=0.35)
ax.set_title("4D high-density points → 3D (PCA), colored by CMYK→RGB")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.view_init(elev=20, azim=35)
pca_path = "cmyk_4d_to_3d_pca.png"
plt.savefig(pca_path, dpi=200, bbox_inches="tight")
plt.close(fig)

sub_n = 4500
sub_idx = rng.choice(len(P_keep), size=sub_n, replace=False)
P_sub = P_keep[sub_idx]; rgb_sub = rgb_keep[sub_idx]
iso = Isomap(n_neighbors=18, n_components=3)
Yiso = iso.fit_transform(P_sub)
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Yiso[:,0], Yiso[:,1], Yiso[:,2], c=rgb_sub, s=3, alpha=0.45)
ax.set_title("4D high-density manifold → 3D (Isomap), colored by CMYK→RGB")
ax.set_xlabel("Iso1"); ax.set_ylabel("Iso2"); ax.set_zlabel("Iso3")
ax.view_init(elev=20, azim=35)
isomap_path = "cmyk_4d_to_3d_isomap.png"
plt.savefig(isomap_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# C) 3D isosurface of 4D density field is exactly the w-fixed isosurface (done above).
# D) 3D projection of 4D object: integrate over w samples
def projected_alpha_xyz(res=64, w_samples=9):
    grid = np.linspace(-1,1,res, dtype=np.float32)
    X,Y,Z = np.meshgrid(grid, grid, grid, indexing="ij")
    ws = np.linspace(-1,1,w_samples, dtype=np.float32)
    acc = np.zeros_like(X, dtype=np.float32)
    for w0 in ws:
        C = class_field_xyz(X,Y,Z,w0, C_centers)
        M = class_field_xyz(X,Y,Z,w0, M_centers)
        Yf= class_field_xyz(X,Y,Z,w0, Y_centers)
        Kk= class_field_xyz(X,Y,Z,w0, K_centers)
        stack = np.stack([C,M,Yf,Kk], axis=0)
        max1 = np.max(stack, axis=0)
        m = stack == max1[None,...]
        stack_masked = np.where(m, -np.inf, stack)
        max2 = np.max(stack_masked, axis=0)
        tie_pen = gelu(-0.9*(max1-max2).astype(np.float32)).astype(np.float32)
        for f in (C,M,Yf,Kk):
            f *= (1 - 0.35 * tie_pen)
        S = C+M+Yf+Kk + 1e-7
        acc += S
    acc = acc / (acc.max() + 1e-9)
    return (X,Y,Z), acc

resP = 64
(Xp,Yp,Zp), proj = projected_alpha_xyz(res=resP, w_samples=9)
levelP = 0.22
vP,fP,_,_ = marching_cubes(proj, level=levelP, spacing=(2/(resP-1),)*3)
vPw = (vP - (resP-1)/2) * (2/(resP-1))
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(vPw[:,0], vPw[:,1], vPw[:,2], triangles=fP, linewidth=0.08, alpha=0.9)
ax.set_title(f"3D projection of 4D object (integrate over w), isosurface level={levelP}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=35)
proj_iso_path = "cmyk_4d_projected_to_3d_isosurface.png"
plt.savefig(proj_iso_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# E) Animate 3D isosurface orbit under (z,w) rotation: vary (z0,w0) and evaluate alpha(x,y,z+z0,w0)
def alpha_xyz_with_zw_shift(z0, w0, res=54):
    grid = np.linspace(-1,1,res, dtype=np.float32)
    X,Y,Z = np.meshgrid(grid, grid, grid, indexing="ij")
    Zs = Z + z0
    C = class_field_xyz(X,Y,Zs,w0, C_centers)
    M = class_field_xyz(X,Y,Zs,w0, M_centers)
    Yf= class_field_xyz(X,Y,Zs,w0, Y_centers)
    Kk= class_field_xyz(X,Y,Zs,w0, K_centers)
    stack = np.stack([C,M,Yf,Kk], axis=0)
    max1 = np.max(stack, axis=0)
    m = stack == max1[None,...]
    stack_masked = np.where(m, -np.inf, stack)
    max2 = np.max(stack_masked, axis=0)
    tie_pen = gelu(-0.9*(max1-max2).astype(np.float32)).astype(np.float32)
    for f in (C,M,Yf,Kk):
        f *= (1 - 0.35 * tie_pen)
    S = C+M+Yf+Kk + 1e-7
    a = np.clip(S / S.max(), 0, 1).astype(np.float32)
    return a

frames = 36
resA = 54
levelA = 0.26
thetas = np.linspace(0, 2*np.pi, frames, endpoint=False)
meshes = []
for th in thetas:
    z0 = float(np.cos(th)*0.35)
    w0 = float(np.sin(th)*0.35)
    a = alpha_xyz_with_zw_shift(z0, w0, res=resA)
    try:
        v,f,_,_ = marching_cubes(a, level=levelA, spacing=(2/(resA-1),)*3)
        vw = (v - (resA-1)/2) * (2/(resA-1))
        meshes.append((vw,f))
    except Exception:
        meshes.append((None,None))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
def upd(i):
    ax.cla()
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=20, azim=35)
    ax.set_title(f"Isosurface orbit under (z,w) rotation\nθ={np.rad2deg(thetas[i]):.0f}°  level={levelA}")
    vw,f = meshes[i]
    if vw is not None:
        ax.plot_trisurf(vw[:,0], vw[:,1], vw[:,2], triangles=f, linewidth=0.05, alpha=0.9)
    return []

ani = FuncAnimation(fig, upd, frames=frames, interval=140)
orbit_iso_gif = "cmyk_3d_isosurface_orbit.gif"
ani.save(orbit_iso_gif, writer=PillowWriter(fps=10))
plt.close(fig)

iso_path, cloud_path, pca_path, isomap_path, proj_iso_path, orbit_iso_gif
