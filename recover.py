"""
Disparity recovery methods for stereo matching.

All functions take BGR images (as returned by cv2.imread) and return a
disparity map of shape (H, W) in pixels, with 0 meaning invalid/border.

Methods:
  recover_disp_sgm         -- OpenCV StereoSGBM baseline
  recover_disp_tv_v2       -- TV-ADMM (Chambolle, D=I formulation)
  recover_disp_learned     -- PnP-ADMM with Gaussian-trained DnCNN (baseline)
  recover_disp_hqs         -- HQS with Gaussian-trained DnCNN (baseline)
  recover_disp_hqs_wta_l1  -- HQS with domain-matched WTA-trained DnCNN (main method)
  recover_disp_sgm_denoised-- SGM followed by learned DnCNN refinement
"""

import numpy as np
import cv2
import torch

from costs import init_cost_census
from model import DnCNN
from utils import tv_denoise

# ---- Lazy-loaded model singletons ----
# Each model is loaded on first use and cached for subsequent calls.

_denoiser_gaussian = None
_denoiser_wta_l1   = None
_denoiser_sgm      = None


def _load_denoiser_gaussian(device):
    """Gaussian-noise DnCNN, trained on clean+Gaussian-noise pairs."""
    global _denoiser_gaussian
    if _denoiser_gaussian is None:
        _denoiser_gaussian = DnCNN(depth=8, channels=64).to(device)
        _denoiser_gaussian.load_state_dict(
            torch.load("models/denoiser_gaussian.pth", map_location=device))
        _denoiser_gaussian.eval()
    return _denoiser_gaussian


def _load_denoiser_wta_l1(device):
    """Domain-matched DnCNN, trained on (WTA output, GT disparity) pairs with L1 loss."""
    global _denoiser_wta_l1
    if _denoiser_wta_l1 is None:
        _denoiser_wta_l1 = DnCNN(depth=8, channels=64).to(device)
        _denoiser_wta_l1.load_state_dict(
            torch.load("models/denoiser_wta_l1.pth", map_location=device))
        _denoiser_wta_l1.eval()
    return _denoiser_wta_l1


def _load_denoiser_sgm(device):
    """SGM post-processing DnCNN, trained on (SGM output, GT disparity) pairs with L1 loss."""
    global _denoiser_sgm
    if _denoiser_sgm is None:
        _denoiser_sgm = DnCNN(depth=8, channels=64).to(device)
        _denoiser_sgm.load_state_dict(
            torch.load("models/denoiser_sgm.pth", map_location=device))
        _denoiser_sgm.eval()
    return _denoiser_sgm


def _run_denoiser(model, d, D_max, device):
    """Normalize d to [0,1], run model forward pass, return in pixel units."""
    inp = np.clip(d / D_max, 0, 1).astype(np.float32)
    t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    return out.squeeze().cpu().numpy() * D_max


def _build_cost_volume(L_img, R_img, D_max, win_size):
    """Compute normalized census cost volume. Returns (cost_full, cost_crop, crop_slices)."""
    L_mono = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    R_mono = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    cost = init_cost_census(L_mono, R_mono, D_max, win_size, census_win=7)
    cost = cost / (7 * 7)  # normalize Hamming distances to [0, 1]

    h, w = L_mono.shape
    r = win_size // 2
    y0, y1 = r, h - r
    x0, x1 = (D_max - 1) + r, w - r
    return cost, cost[y0:y1, x0:x1], (y0, y1, x0, x1)


def _hqs_d_update(cost_c, z, mu, D_max):
    """HQS d-update: argmin_d C(d) + (mu/2)||d - z||^2, solved per-pixel."""
    D_vals = np.arange(D_max, dtype=np.float32).reshape(1, 1, -1)
    target = z[:, :, None]
    return np.argmin(cost_c + (mu / 2) * (D_vals - target) ** 2, axis=2).astype(np.float32)


# ---- Public recovery functions ----

def recover_disp_sgm(L_img, R_img, D_max=192, win_size=11):
    """SGM baseline: OpenCV StereoSGBM with 8-direction dynamic programming."""
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=D_max,
        blockSize=win_size,
        P1=8  * win_size ** 2,
        P2=32 * win_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(L_img, R_img).astype(np.float32) / 16.0
    disp[disp < 0] = 0  # invalid pixels come out as -1
    return disp


def recover_disp_tv_v2(L_img, R_img, mu, lam, D_max, win_size, iters=20):
    """TV-ADMM with D=I formulation (Chambolle's algorithm for z-update).

    Minimizes: sum_uv C(u,v,d(u,v)) + lam * TV(d)

    ADMM splitting z = d:
      d-update: argmin_d C(d) + (mu/2)||d - (z-u)||^2  -- solved per-pixel
      z-update: TV denoising of (d+u) via Chambolle's algorithm
      u-update: u = u + d - z
    """
    cost, cost_c, (y0, y1, x0, x1) = _build_cost_volume(L_img, R_img, D_max, win_size)
    D_vals    = np.arange(D_max, dtype=np.float32).reshape(1, 1, -1)
    d         = np.argmin(cost_c, axis=2).astype(np.float32)  # WTA init
    z         = d.copy()
    u         = np.zeros_like(d)
    threshold = lam / mu  # effective TV weight for Chambolle

    for _ in range(iters):
        target = (z - u)[:, :, None]
        d = np.argmin(cost_c + (mu / 2) * (D_vals - target) ** 2, axis=2).astype(np.float32)
        z = tv_denoise(d + u, threshold)
        u = u + d - z

    d_full = np.argmin(cost, axis=2).astype(np.float32)
    d_full[y0:y1, x0:x1] = d
    return d_full


def recover_disp_learned(L_img, R_img, mu, D_max, win_size, iters=20):
    """PnP-ADMM with Gaussian-trained DnCNN as learned prior (baseline comparison).

    Uses the Gaussian denoiser as a plug-and-play prior. Tends to diverge because
    the dual variable u accumulates, pushing (d+u) out of the denoiser's training
    distribution [0, D_max].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_denoiser_gaussian(device)

    cost, cost_c, (y0, y1, x0, x1) = _build_cost_volume(L_img, R_img, D_max, win_size)
    D_vals = np.arange(D_max, dtype=np.float32).reshape(1, 1, -1)
    d = np.argmin(cost_c, axis=2).astype(np.float32)
    z = d.copy()
    u = np.zeros_like(d)

    for _ in range(iters):
        target = (z - u)[:, :, None]
        d = np.argmin(cost_c + (mu / 2) * (D_vals - target) ** 2, axis=2).astype(np.float32)
        z = _run_denoiser(model, np.clip(d + u, 0, D_max), D_max, device)
        u = u + d - z

    d_full = np.argmin(cost, axis=2).astype(np.float32)
    d_full[y0:y1, x0:x1] = d
    return d_full


def recover_disp_hqs(L_img, R_img, mu, D_max, win_size, iters=20):
    """HQS with Gaussian-trained DnCNN (baseline comparison).

    Half-Quadratic Splitting removes the dual variable u entirely.
    The denoiser always sees d directly (more stable than PnP-ADMM), but the
    Gaussian denoiser is mismatched to the WTA cost-volume input distribution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_denoiser_gaussian(device)

    cost, cost_c, (y0, y1, x0, x1) = _build_cost_volume(L_img, R_img, D_max, win_size)
    z = np.argmin(cost_c, axis=2).astype(np.float32)  # WTA init

    for _ in range(iters):
        d = _hqs_d_update(cost_c, z, mu, D_max)
        z = _run_denoiser(model, d, D_max, device)

    d_full = np.argmin(cost, axis=2).astype(np.float32)
    d_full[y0:y1, x0:x1] = d
    return d_full


def recover_disp_hqs_wta_l1(L_img, R_img, mu, D_max, win_size, iters=20):
    """HQS with domain-matched WTA-trained DnCNN (main method).

    Trained on (WTA output, GT disparity) pairs from vKITTI with L1 loss,
    so the denoiser input distribution matches exactly what it sees at inference.
    mu=0.01 recommended (see ablation in paper).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_denoiser_wta_l1(device)

    cost, cost_c, (y0, y1, x0, x1) = _build_cost_volume(L_img, R_img, D_max, win_size)
    z = np.argmin(cost_c, axis=2).astype(np.float32)  # WTA init

    for _ in range(iters):
        d = _hqs_d_update(cost_c, z, mu, D_max)
        z = _run_denoiser(model, d, D_max, device)

    d_full = np.argmin(cost, axis=2).astype(np.float32)
    d_full[y0:y1, x0:x1] = d
    return d_full


def recover_disp_sgm_denoised(L_img, R_img, D_max, win_size):
    """SGM followed by a single-pass DnCNN refinement (trained on SGM->GT pairs)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_denoiser_sgm(device)
    sgm    = recover_disp_sgm(L_img, R_img, D_max, win_size)
    return _run_denoiser(model, sgm, D_max, device)
