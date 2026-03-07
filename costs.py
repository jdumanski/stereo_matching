import numpy as np
from functools import lru_cache
import cv2

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
