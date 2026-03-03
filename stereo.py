import numpy as np
from functools import lru_cache
import cv2
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

@lru_cache(maxsize=None)
def gaussian_kernel(w, sigma):
    ax = np.arange(-(w//2), w//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)   # normalize
    return kernel.astype(np.float32)

def init_cost(L, R, D_max, win_size):
    h = L.shape[0]
    w = L.shape[1]
    sigma = 1.0
    g = gaussian_kernel(win_size, sigma)
    cf = np.full((h, w, D_max), np.inf, dtype=np.float32)
    r = win_size // 2
    y0, y1 = r, h - r
    x0, x1 = (D_max - 1) + r, w - r
    for d in range(D_max):
        R_shift = np.zeros_like(R)
        R_shift[:, d:] = R[:,:w-d] # shift right image to right by d to "align" w left image for this disp val d
        diff = np.square(L - R_shift)
        cost_d = cv2.filter2D(diff, ddepth=-1, kernel=g, borderType=cv2.BORDER_CONSTANT) # cost function at curr d
        cf[y0:y1, x0:x1, d] = cost_d[y0:y1, x0:x1] # valid range of pixels
    return cf

def census_transform(img_u8, win=7):
    """
    Compute Census transform as a uint64 descriptor per pixel.
    img_u8: (H,W) uint8
    win: odd window size (<= 9 recommended for 64-bit packing)
    Returns: (H,W) uint64 descriptors. Borders are 0.
    """
    assert win % 2 == 1
    H, W = img_u8.shape
    r = win // 2

    # We'll pack bits into uint64. win=7 => 48 bits (49-1), fits.
    desc = np.zeros((H, W), dtype=np.uint64)
    center = img_u8

    bit = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue

            # valid region where both center and shifted neighbor exist
            y0 = max(0, -dy)
            y1 = min(H, H - dy)
            x0 = max(0, -dx)
            x1 = min(W, W - dx)

            neigh = center[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            cen   = center[y0:y1,         x0:x1]

            # bit = 1 if neighbor < center (you can flip sign, doesn't matter as long as consistent)
            b = (neigh < cen).astype(np.uint64)

            desc[y0:y1, x0:x1] |= (b << bit)
            bit += 1

    # Zero-out the census border where comparisons are incomplete
    desc[:r, :] = 0
    desc[-r:, :] = 0
    desc[:, :r] = 0
    desc[:, -r:] = 0
    return desc

def init_cost_census(L_mono_float, R_mono_float, D_max, win_size, census_win=7, shift_right=True):
    L_u8 = np.clip(L_mono_float * 255.0, 0, 255).astype(np.uint8)
    R_u8 = np.clip(R_mono_float * 255.0, 0, 255).astype(np.uint8)

    H, W = L_u8.shape
    cf = np.full((H, W, D_max), np.inf, dtype=np.float32)

    # Census descriptors
    Lc = census_transform(L_u8, win=census_win)
    Rc = census_transform(R_u8, win=census_win)

    # Use SAME aggregation window as SAD
    g = gaussian_kernel(win_size, sigma=1)

    r = max(win_size // 2, census_win // 2)
    y0, y1 = r, H - r
    x0, x1 = (D_max - 1) + r, W - r

    # popcount lookup (your guaranteed version)
    _POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    for d in range(D_max):
        Rc_shift = np.zeros_like(Rc)

        if shift_right:
            # aligns with using R(x-d): Rc_shift[:, d:] = Rc[:, :W-d]
            if d == 0:
                Rc_shift[:] = Rc
            else:
                Rc_shift[:, d:] = Rc[:, :W-d]
        else:
            # aligns with using R(x+d): Rc_shift[:, :W-d] = Rc[:, d:]
            if d == 0:
                Rc_shift[:] = Rc
            else:
                Rc_shift[:, :W-d] = Rc[:, d:]

        xor = np.bitwise_xor(Lc, Rc_shift)
        xor_u8 = xor.view(np.uint8).reshape(xor.shape + (8,))
        ham = _POPCOUNT8[xor_u8].sum(axis=-1).astype(np.float32)

        cost_d = cv2.filter2D(ham, ddepth=-1, kernel=g, borderType=cv2.BORDER_CONSTANT)
        cf[y0:y1, x0:x1, d] = cost_d[y0:y1, x0:x1]

    return cf

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

def grad(d):
    # fwd diff
    dx = np.zeros_like(d)
    dy = np.zeros_like(d)
    dx[:, :-1] = d[:, 1:] - d[:, :-1]
    dy[:-1, :] = d[1:, :] - d[:-1, :]
    return dx, dy

def div(dx, dy):
    # adj (transpose) of forward grad
    out = np.zeros_like(dx)
    out[:, :-1] -= dx[:, :-1]
    out[:, 1:]  += dx[:, :-1]
    out[:-1, :] -= dy[:-1, :]
    out[1:,  :] += dy[:-1, :]
    return out

# RHS mat for d update LHS formulation
def create_A(mu, shape):
    H, W = shape
    N = H * W
    def matvec(d_flat):
        d = d_flat.reshape(H, W)
        gx, gy = grad(d)
        out = d + mu * div(gx, gy) 
        return out.ravel()
    return LinearOperator((N, N), matvec=matvec, dtype=np.float32)

def soft(a, t):
    return np.sign(a) * np.maximum(np.abs(a)-t, 0)

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

def uniqueness_mask_from_cost(cost, method="ratio", thr=1.3, eps=1e-6):
    """
    cost: (H,W,D) lower is better
    returns mask (H,W) True where unique
    """
    two = np.partition(cost, kth=1, axis=2)[:, :, :2]   # two smallest (unordered)
    two.sort(axis=2)
    best = two[..., 0]
    second = two[..., 1]

    if method == "ratio":
        return (second / (best + eps)) >= thr
    elif method == "margin":
        return (second - best) >= thr
    else:
        raise ValueError("method must be 'ratio' or 'margin'")

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

def main():
    L_img = cv2.imread("000114_l.png")
    R_img = cv2.imread("000114_r.png")
    mu = 1.0
    lam = 100
    D_max = 192
    win_size = 11 # for block similarity
    # disp_map = recover_disp_learned(L_img, R_img, mu, D_max, win_size)
    disp_map = recover_disp_tv(L_img, R_img, mu, lam, D_max, win_size)
    #dm, m = recover_disp_lr_only(L_img, R_img, D_max, win_size, use_uniqueness=False, uniq_method="ratio", uniq_thr=1.3)
    #plt.imshow(dm, cmap='gray')
    #plt.show()
    #print(disp_map.shape)
    plt.imshow(disp_map, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()