import numpy as np
from functools import lru_cache
import cv2

@lru_cache(maxsize=None)
def gaussian_kernel(w, sigma):
    ax = np.arange(-(w//2), w//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)   # normalize
    return kernel

def init_cost(L, R, D_max):
    h = L.shape[0]
    w = L.shape[1]
    win_size = 7
    sigma = 1
    g = gaussian_kernel(win_size, sigma)
    cf = np.full((h, w, D_max), np.inf, dtype=np.float32)
    r = win_size // 2
    y0, y1 = r, h - r
    x0, x1 = (D_max - 1) + r, w - r
    for d in range(D_max):
        R_shift = np.zeros_like(R)
        R_shift[:, d:] = R[:,:w-d] # shift right image to right by d to "align" w left image for this disp val d
        diff = np.abs(L - R_shift)
        cost_d = cv2.filter2D(diff, ddepth=-1, kernel=g, borderType=cv2.BORDER_CONSTANT) # cost function at curr d
        cf[y0:y1, x0:x1, d] = cost_d[y0:y1, x0:x1] # valid range of pixels
    return cf

def recover_disp(L_img, R_img, mu0, lam, alpha, D_max, iters=20):
    L_mono = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    R_mono = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    cost = init_cost(L_mono, R_mono, D_max)
    z = np.argmin(cost, axis=2).astype(np.float32) # initialize z0
    d = None
    D = np.arange(D_max, dtype=np.float32).reshape(1, 1, -1) # list of all possible disp values
    mu = mu0
    for i in range(iters):
        # d update
        d = np.argmin(cost + mu / 2 * (D - z[:,:,None])**2, axis=2)
        # z update
        z = model(d, L_img) # TODO: implement regularizer model
        mu *= alpha
    return d

def main():
    L_img = cv2.imread("training/image_2/000000_10.png")
    R_img = cv2.imread("training/image_3/000000_10.png")
    mu0 = 1
    alpha = 1.25 # multiplier for mu
    lam = 1
    D_max = 192
    recover_disp(L_img, R_img, mu0, lam, alpha, D_max)

if __name__ == "__main__":
    main()