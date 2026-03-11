"""
Train a DnCNN disparity denoiser on (SGM_disparity, GT_disparity) pairs from vKITTI.

The denoiser learns to refine SGM output errors as a post-processing step.
Uses L1 loss. Scene20 is held out from training and used as the test set.

Usage:
    python training/train_sgm_denoiser.py

Saves best val checkpoint to models/denoiser_sgm.pth.
"""

import glob
import random
import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DnCNN

# ---- Hyperparameters ----
VKITTI_RGB_ROOT   = "vkitti_2.0.3_rgb"
VKITTI_DEPTH_ROOT = "vkitti_2.0.3_depth"
D_MAX             = 192.0
F_B               = 725.0087 * 0.532725   # focal_length * baseline (px * m)
WIN_SIZE          = 11
PATCH_SIZE        = 64
PATCHES_PER_IMG   = 8
TRAIN_FRAC        = 0.9
BATCH_SIZE        = 32
EPOCHS            = 60
LR                = 1e-3
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH       = "models/denoiser_sgm.pth"


def load_gt_disp(depth_path):
    """Load vKITTI depth PNG (uint16 in cm) -> disparity map in [0, D_MAX]."""
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid = raw < 65535  # 65535 = invalid
    disp = np.zeros_like(raw)
    disp[valid] = np.clip(F_B / (raw[valid] / 100.0), 0, D_MAX)
    return disp, valid


def compute_sgm(rgb_left_path, rgb_right_path):
    """Compute SGM disparity using OpenCV StereoSGBM."""
    L = cv2.imread(rgb_left_path)
    R = cv2.imread(rgb_right_path)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=int(D_MAX),
        blockSize=WIN_SIZE,
        P1=8  * WIN_SIZE ** 2,
        P2=32 * WIN_SIZE ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(L, R).astype(np.float32) / 16.0
    disp[disp < 0] = 0
    return disp


def build_cache(pairs):
    """Precompute all (sgm, gt) pairs and store in RAM."""
    print(f"Precomputing SGM for {len(pairs)} images...")
    cache = []
    for i, (rgb_l, rgb_r, depth) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}")
        sgm = compute_sgm(rgb_l, rgb_r)
        gt, valid = load_gt_disp(depth)
        cache.append((sgm, gt, valid))
    return cache


class SGMDataset(Dataset):
    def __init__(self, cache, patch_size=PATCH_SIZE, patches_per_img=PATCHES_PER_IMG):
        self.cache = cache
        self.P = patch_size
        self.K = patches_per_img

    def __len__(self):
        return len(self.cache) * self.K

    def __getitem__(self, idx):
        sgm, gt, valid = self.cache[idx // self.K]
        H, W = gt.shape
        P = self.P

        # Sample a patch where >70% of GT pixels are valid
        for _ in range(20):
            y = random.randint(0, H - P)
            x = random.randint(0, W - P)
            if valid[y:y+P, x:x+P].mean() > 0.7:
                break

        sgm_patch = (sgm[y:y+P, x:x+P] / D_MAX).astype(np.float32)
        gt_patch  = (gt[y:y+P,  x:x+P] / D_MAX).astype(np.float32)

        return (torch.from_numpy(sgm_patch).unsqueeze(0),
                torch.from_numpy(gt_patch).unsqueeze(0))


def main():
    print(f"Using device: {DEVICE}")

    # Collect all clone-condition stereo pairs, hold out Scene20 for testing
    left_paths = sorted(glob.glob(f"{VKITTI_RGB_ROOT}/*/clone/frames/rgb/Camera_0/*.jpg"))
    left_paths = [p for p in left_paths if "/Scene20/" not in p]
    print(f"Found {len(left_paths)} training stereo pairs (Scene20 held out)")

    pairs = []
    for lp in left_paths:
        rp = lp.replace("Camera_0", "Camera_1")
        dp = lp.replace(VKITTI_RGB_ROOT, VKITTI_DEPTH_ROOT) \
               .replace("rgb/Camera_0/rgb_", "depth/Camera_0/depth_") \
               .replace(".jpg", ".png")
        if glob.glob(rp) and glob.glob(dp):
            pairs.append((lp, rp, dp))
    print(f"Matched {len(pairs)} valid triplets (left, right, depth)")

    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_FRAC)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    train_cache = build_cache(train_pairs)
    val_cache   = build_cache(val_pairs)

    train_ds = SGMDataset(train_cache)
    val_ds   = SGMDataset(val_cache, patches_per_img=4)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = DnCNN(depth=8, channels=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.L1Loss()

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            pred = model(noisy)
            loss = criterion(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                val_loss += criterion(model(noisy), clean).item() * noisy.size(0)
        val_loss /= len(val_ds)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | train MAE {train_loss * D_MAX:.3f}px | val MAE {val_loss * D_MAX:.3f}px")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  -> saved {OUTPUT_PATH}")

    print(f"Done. Best val MAE: {best_val * D_MAX:.3f} px")


if __name__ == "__main__":
    main()
