import numpy as np

import matplotlib.pyplot as plt

from scipy.special import erf

import time, json, os



# ================= CONFIG (updated for 10 slices) =================

RES_HI = 420

RES_COARSE = 56

SHARPNESS = 2.8

TIE_GAMMA = 0.9

TIE_STRENGTH = 0.35

INTENSITY_SCALE = True



Z0_RANGE = (-0.4, 0.4); Z0_STEPS = 5

W0_RANGE = (-0.4, 0.4); W0_STEPS = 5

SLOPES = np.array([-0.4, 0.0, 0.4], dtype=np.float32)



ROT_BASE_DEG = 10.0      # step size for the fan

NUM_ROTATED = 10         # produce 10 rotated slices

SEED = 7



C_centers = [np.array([ 0.5,  0.4, -0.2,  0.3], dtype=np.float32)]

M_centers = [np.array([-0.6,  0.1,  0.6, -0.4], dtype=np.float32)]

Y_centers = [np.array([ 0.1, -0.5, -0.4,  0.5], dtype=np.float32)]

K_centers = [np.array([-0.2, -0.3,  0.5, -0.6], dtype=np.float32)]

CLASSES = [C_centers, M_centers, Y_centers, K_centers]



def gelu(x): return 0.5 * x * (1 + erf(x / np.sqrt(2)))



def eval_slice_affine(res, o, a, b, sharpness=SHARPNESS, tie_gamma=TIE_GAMMA, tie_strength=TIE_STRENGTH, intensity_scale=INTENSITY_SCALE):

    u = np.linspace(-1, 1, res, dtype=np.float32); v = np.linspace(-1, 1, res, dtype=np.float32)

    U, V = np.meshgrid(u, v, indexing="ij")

    X = o[0] + U*a[0] + V*b[0]; Y = o[1] + U*a[1] + V*b[1]; Z = o[2] + U*a[2] + V*b[2]; W = o[3] + U*a[3] + V*b[3]

    fields = []

    for centers in CLASSES:

        f = np.zeros_like(X, dtype=np.float32)

        for c in centers:

            dx = X - c[0]; dy = Y - c[1]; dz = Z - c[2]; dw = W - c[3]

            d = np.sqrt(dx*dx + dy*dy + dz*dz + dw*dw, dtype=np.float32)

            f += gelu((1.0 - d) * sharpness).astype(np.float32)

        fields.append(f)

    fields = np.stack(fields, axis=0)

    max1 = np.max(fields, axis=0)

    neg_inf = np.full_like(fields, -np.inf, dtype=np.float32)

    mask = (fields == max1[None, ...])

    fields_masked = np.where(mask, neg_inf, fields); max2 = np.max(fields_masked, axis=0)

    tie_pen = gelu(-tie_gamma * (max1 - max2).astype(np.float32)).astype(np.float32)

    fields *= (1 - tie_strength * tie_pen)[None, ...]

    S = fields.sum(axis=0) + np.float32(1e-7)

    w = fields / S[None, ...]

    wC, wM, wY, wK = w[0], w[1], w[2], w[3]

    R = (1 - wM) * (1 - wK); G = (1 - wY) * (1 - wK); B = (1 - wC) * (1 - wK)

    if intensity_scale:

        intensity = np.clip(S / S.max(), 0, 1).astype(np.float32); R *= intensity; G *= intensity; B *= intensity

    RGB = np.clip(np.stack([R, G, B], axis=-1), 0, 1).astype(np.float32)

    return RGB, fields, S



def score_float32(RGB, S):

    act = float(S.mean()); var = float(np.var(RGB.reshape(-1,3), axis=0).mean())

    return 0.6*act + 0.4*var



def coarse_int8_search(res=RES_COARSE):

    z0_vals = np.linspace(Z0_RANGE[0], Z0_RANGE[1], Z0_STEPS, dtype=np.float32)

    w0_vals = np.linspace(W0_RANGE[0], W0_RANGE[1], W0_STEPS, dtype=np.float32)

    slopes = SLOPES

    best=None; best_params=None

    for z0 in z0_vals:

        for w0 in w0_vals:

            for sz_u in slopes:

                for sw_u in slopes:

                    for sz_v in slopes:

                        for sw_v in slopes:

                            o = np.array([0.0, 0.0, z0, w0], dtype=np.float32)

                            a = np.array([1.0, 0.0, sz_u, sw_u], dtype=np.float32)

                            b = np.array([0.0, 1.0, sz_v, sw_v], dtype=np.float32)

                            RGB, fields, S = eval_slice_affine(res, o, a, b)

                            fmin, fmax = fields.min(), fields.max()

                            if fmax <= fmin + 1e-8: continue

                            fields_u8 = np.clip(((fields - fmin) / (fmax - fmin) * 255.0).round(), 0, 255).astype(np.uint8)

                            S_u8 = np.clip(fields_u8.sum(axis=0), 0, 255).astype(np.uint8)

                            fields32 = fields_u8.astype(np.float32); S32 = S_u8.astype(np.float32) + 1e-7

                            w = fields32 / S32[None, ...]

                            wC, wM, wY, wK = w[0], w[1], w[2], w[3]

                            R = (1 - wM) * (1 - wK); G = (1 - wY) * (1 - wK); B = (1 - wC) * (1 - wK)

                            RGBu = np.clip(np.stack([R, G, B], axis=-1), 0, 1).astype(np.float32)

                            act = float(S_u8.mean()) / 255.0; var = float(np.var(RGBu.reshape(-1,3), axis=0).mean())

                            sc = 0.6*act + 0.4*var

                            if (best is None) or (sc > best):

                                best = sc; best_params = (o, a, b)

    dz = (Z0_RANGE[1] - Z0_RANGE[0]) / (Z0_STEPS - 1)

    dw = (W0_RANGE[1] - W0_RANGE[0]) / (W0_STEPS - 1)

    ds = (SLOPES[1] - SLOPES[0]) if len(SLOPES) > 1 else 0.0

    return best_params, (dz/2.0, dw/2.0, ds/2.0)



def orthonormalize(a, b, eps=1e-8):

    a = a.astype(np.float32); b = b.astype(np.float32)

    na = np.linalg.norm(a) + eps; a /= na

    b = b - (a @ b) * a; nb = np.linalg.norm(b) + eps; b /= nb

    return a, b



def pick_perp_axis(a, b, seed=SEED):

    rng = np.random.default_rng(seed); v = rng.normal(size=a.shape).astype(np.float32)

    a1, b1 = orthonormalize(a, b); v = v - (v @ a1) * a1 - (v @ b1) * b1

    nv = np.linalg.norm(v) + 1e-8; return v / nv



def rotate_plane(o, a, b, axis_perp, angle_deg):

    a1, b1 = orthonormalize(a, b)

    n = axis_perp.copy(); n = n - (n @ a1) * a1 - (n @ b1) * b1

    n /= (np.linalg.norm(n) + 1e-8)

    theta = np.deg2rad(angle_deg).astype(np.float32)

    a_rot = (np.cos(theta) * a1) + (np.sin(theta) * n)

    a_rot, b_new = orthonormalize(a_rot, b1); return o, a_rot, b_new



def main(
    output_dir="/mnt/data",
    res_hi=RES_HI,
    res_coarse=RES_COARSE,
    num_rotated=NUM_ROTATED,
    rot_base_deg=ROT_BASE_DEG,
    z0_steps=Z0_STEPS,
    w0_steps=W0_STEPS,
    slopes=SLOPES,
    seed=SEED,
):
    global RES_HI, RES_COARSE, NUM_ROTATED, ROT_BASE_DEG, Z0_STEPS, W0_STEPS, SLOPES, SEED
    old_vals = (
        RES_HI,
        RES_COARSE,
        NUM_ROTATED,
        ROT_BASE_DEG,
        Z0_STEPS,
        W0_STEPS,
        SLOPES,
        SEED,
    )
    RES_HI, RES_COARSE, NUM_ROTATED, ROT_BASE_DEG, Z0_STEPS, W0_STEPS, SLOPES, SEED = (
        res_hi,
        res_coarse,
        num_rotated,
        rot_base_deg,
        z0_steps,
        w0_steps,
        slopes,
        seed,
    )
    try:
        t0 = time.time()
        (best_o, best_a, best_b), (dz2, dw2, ds2) = coarse_int8_search(res=RES_COARSE)
        t1 = time.time()

        o_low = best_o.copy(); o_low[2] -= dz2; o_low[3] -= dw2
        a_low = best_a.copy(); a_low[2] -= ds2; a_low[3] -= ds2
        b_low = best_b.copy(); b_low[2] -= ds2; b_low[3] -= ds2

        o_high = best_o.copy(); o_high[2] += dz2; o_high[3] += dw2
        a_high = best_a.copy(); a_high[2] += ds2; a_high[3] += ds2
        b_high = best_b.copy(); b_high[2] += ds2; b_high[3] += ds2

        RGB_low, F_low, S_low = eval_slice_affine(RES_HI, o_low, a_low, b_low)
        sc_low = score_float32(RGB_low, S_low)
        RGB_high, F_high, S_high = eval_slice_affine(RES_HI, o_high, a_high, b_high)
        sc_high = score_float32(RGB_high, S_high)

        if sc_high >= sc_low:
            label, o0, a0, b0, RGB0 = "upper", o_high, a_high, b_high, RGB_high
        else:
            label, o0, a0, b0, RGB0 = "lower", o_low, a_low, b_low, RGB_low
        t2 = time.time()

        axis_perp = pick_perp_axis(a0, b0, seed=SEED)

        half = NUM_ROTATED // 2
        angles = [ROT_BASE_DEG * (i - half) for i in range(NUM_ROTATED)]

        RGBc, Fc, Sc = eval_slice_affine(RES_COARSE, best_o, best_a, best_b)
        dens_map = (Sc / (Sc.max() + 1e-7)).astype(np.float32)
        os.makedirs(output_dir, exist_ok=True)
        coarse_path = os.path.join(output_dir, "coarse_density_map.png")
        plt.imsave(coarse_path, dens_map, cmap="gray")

        base_path = os.path.join(
            output_dir,
            f"slice_origin_{label}_z{float(o0[2]):+.3f}_w{float(o0[3]):+.3f}.png",
        )
        plt.imsave(base_path, RGB0)
        paths = {"origin": base_path, "coarse_density": coarse_path}

        rot_paths = []
        for ang in angles:
            o_r, a_r, b_r = rotate_plane(o0, a0, b0, axis_perp, ang)
            RGB_r, _, _ = eval_slice_affine(RES_HI, o_r, a_r, b_r)
            pth = os.path.join(output_dir, f"slice_rot_{int(ang):+d}deg.png")
            plt.imsave(pth, RGB_r)
            rot_paths.append(pth)
            paths[f"rot_{ang:+.1f}"] = pth

        t3 = time.time()
        summary = {
            "timings_s": {
                "coarse_search": t1 - t0,
                "refine": t2 - t1,
                "rotations": t3 - t2,
            },
            "best_params_int8": {
                "o": best_o.tolist(),
                "a": best_a.tolist(),
                "b": best_b.tolist(),
                "half_steps": {
                    "dz2": float(dz2),
                    "dw2": float(dw2),
                    "ds2": float(ds2),
                },
            },
            "chosen_origin": {
                "which_bound": label,
                "o": o0.tolist(),
                "a": a0.tolist(),
                "b": b0.tolist(),
            },
            "rotation_angles_deg": angles,
            "paths": paths,
        }
        return summary
    finally:
        (
            RES_HI,
            RES_COARSE,
            NUM_ROTATED,
            ROT_BASE_DEG,
            Z0_STEPS,
            W0_STEPS,
            SLOPES,
            SEED,
        ) = old_vals


if __name__ == "__main__":
    main()
