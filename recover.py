import numpy as np
import cv2
from scipy.sparse.linalg import cg

from costs import init_cost, init_cost_census
from utils import grad, div, soft, create_A, uniqueness_mask_from_cost

def recover_disp_learned(L_img, R_img, mu, D_max, win_size, iters=20):
    L_mono = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    R_mono = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    cost = init_cost(L_mono, R_mono, D_max, 7)

    h, w = L_mono.shape
    r = win_size // 2
    y0, y1 = r, h - r
    x0, x1 = (D_max - 1) + r, w - r
    cost_c = cost[y0:y1, x0:x1] # only perform admm on valid region! shape (Hv, Wv, D_max)

    z = np.argmin(cost_c, axis=2).astype(np.float32)
    u = np.zeros_like(z, dtype=np.float32)
    D = np.arange(D_max, dtype=np.float32).reshape(1, 1, -1) # list of all possible disp values
    # ADMM w learned prior
    for i in range(iters):
        # d update
        d = np.argmin(cost_c + (mu / 2) * (D - (z-u)[:,:,None])**2, axis=2).astype(np.float32)
        # z update
        z = model(d+u, L_img) # TODO: implement regularizer model. Note it must output img of size (Hv, Wv) (or crop)
        # u update
        u = u + d - z
    d_full = np.argmin(cost, axis=2).astype(np.float32)
    d_full[y0:y1, x0:x1] = d
    return d_full

def recover_disp_tv(L_img, R_img, mu, lam, D_max, win_size, iters=20):
    L_mono = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    R_mono = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    #cost = init_cost(L_mono, R_mono, D_max, win_size)
    cost = init_cost_census(L_mono, R_mono, D_max, win_size, census_win=7)

    #plt.imshow(d_wta, cmap="gray"); plt.show()
    h, w = L_mono.shape
    r = win_size // 2
    y0, y1 = r, h - r
    x0, x1 = (D_max - 1) + r, w - r

    d_data_full = np.argmin(cost, axis=2).astype(np.float32)
    d_data = d_data_full[y0:y1, x0:x1] # only perform admm on valid region!
    d = d_data.copy()
    dx, dy = grad(d)
    z = [dx, dy] # (zx, zy)
    u = [np.zeros_like(d, dtype=np.float32), np.zeros_like(d, dtype=np.float32)] # (ux, uy)
    A = create_A(mu, d.shape)
    k = lam / mu
    for i in range(iters):
        # d update
        b = d_data + mu * div(z[0]-u[0], z[1]-u[1])
        d_flat, _ = cg(A, b.ravel(), x0=d.ravel(), maxiter=50, rtol=1e-6)
        d = d_flat.reshape(d.shape).astype(np.float32)
        # z update
        dx, dy = grad(d)
        vx = dx + u[0]
        vy = dy + u[1]
        z[0] = soft(vx, k)
        z[1] = soft(vy, k)
        # u update
        u[0] = u[0] + dx - z[0]
        u[1] = u[1] + dy - z[1]
    d_out = d_data_full.copy()
    d_out[y0:y1, x0:x1] = d
    return d_out

def recover_disp_lr_only(L_img, R_img, D_max, win_size, census_win=7, tau=1,
                         use_uniqueness=True, uniq_thr=1.3, uniq_method="ratio"):
    L_mono = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    R_mono = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    cost_L = init_cost(L_mono, R_mono, D_max, win_size, census_win, shift_right=True)
    cost_R = init_cost(R_mono, L_mono, D_max, win_size, census_win, shift_right=False)

    dL = np.argmin(cost_L, axis=2).astype(np.int32)
    dR = np.argmin(cost_R, axis=2).astype(np.int32)

    H, W = dL.shape
    r = max(win_size // 2, census_win // 2)
    y0, y1 = r, H - r
    x0, x1 = (D_max - 1) + r, W - r

    # --- LR consistency mask on crop ---
    mask_lr = np.zeros((H, W), dtype=bool)
    ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    xr = xs - dL[ys, xs]
    in_bounds = (xr >= 0) & (xr < W)

    ys_v = ys[in_bounds]
    xs_v = xs[in_bounds]
    xr_v = xr[in_bounds]

    dR_at_match = dR[ys_v, xr_v]
    ok = np.abs(dL[ys_v, xs_v] - dR_at_match) <= tau
    mask_lr[ys_v[ok], xs_v[ok]] = True

    # --- Uniqueness mask from LEFT cost volume (on same crop) ---
    if use_uniqueness:
        mask_u_crop = uniqueness_mask_from_cost(
            cost_L[y0:y1, x0:x1, :],
            method=uniq_method,
            thr=uniq_thr
        )
        mask_u = np.zeros((H, W), dtype=bool)
        mask_u[y0:y1, x0:x1] = mask_u_crop
        mask = mask_lr & mask_u
    else:
        mask = mask_lr

    d_masked = dL.astype(np.float32)
    d_masked[~mask] = 0.0
    return d_masked, mask
