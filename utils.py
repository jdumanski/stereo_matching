import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

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

def tv_denoise(v, threshold, iters=20):
    """Chambolle's algorithm: proximal operator of TV, i.e. argmin_z (lam/mu)*TV(z) + 0.5*||z-v||^2.
    threshold = lam / mu (the effective TV weight)."""
    p = [np.zeros_like(v), np.zeros_like(v)]
    tau = 0.125
    for _ in range(iters):
        z = v - div(p[0], p[1])
        gx, gy = grad(z)
        p[0] = np.clip(p[0] + tau * gx, -threshold, threshold)
        p[1] = np.clip(p[1] + tau * gy, -threshold, threshold)
    return v - div(p[0], p[1])

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
