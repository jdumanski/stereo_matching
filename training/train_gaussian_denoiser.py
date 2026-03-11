"""
Train a DnCNN denoiser on vKITTI GT disparity maps with additive Gaussian noise.

This is the baseline denoiser used to demonstrate domain mismatch: the model is
trained on (clean + Gaussian noise) pairs, but at inference time it receives WTA
cost-volume outputs — a completely different input distribution — causing it to fail.

Usage:
    python training/train_gaussian_denoiser.py

Saves best val checkpoint to models/denoiser_gaussian.pth.
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
VKITTI_DEPTH_ROOT = "vkitti_2.0.3_depth"
D_MAX             = 192.0
F_B               = 725.0087 * 0.532725   # focal_length * baseline (px * m)
PATCH_SIZE        = 64
PATCHES_PER_IMG   = 16       # random crops per image
NOISE_MIN         = 2.0      # Gaussian noise std range (disparity pixels)
NOISE_MAX         = 25.0
TRAIN_FRAC        = 0.9
BATCH_SIZE        = 64
EPOCHS            = 30
LR                = 1e-3
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH       = "models/denoiser_gaussian.pth"


def load_disp_from_depth(path):
    """Load vKITTI depth PNG (uint16 in cm) -> disparity map in [0, D_MAX]."""
    raw   = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid = raw < 65535  # 65535 = invalid
    disp  = np.zeros_like(raw)
    disp[valid] = np.clip(F_B / (raw[valid] / 100.0), 0, D_MAX)
    return disp, valid


class DispDataset(Dataset):
    def __init__(self, paths, patch_size=PATCH_SIZE, patches_per_img=PATCHES_PER_IMG):
        self.paths = paths
        self.P     = patch_size
        self.K     = patches_per_img

    def __len__(self):
        return len(self.paths) * self.K

    def __getitem__(self, idx):
        disp, valid = load_disp_from_depth(self.paths[idx // self.K])
        H, W = disp.shape
        P    = self.P

        # Sample a patch where >70% of GT pixels are valid
        for _ in range(20):
            y = random.randint(0, H - P)
            x = random.randint(0, W - P)
            if valid[y:y+P, x:x+P].mean() > 0.7:
                break

        # Normalize to [0, 1], add Gaussian noise with random sigma
        clean = (disp[y:y+P, x:x+P] / D_MAX).astype(np.float32)
        sigma = random.uniform(NOISE_MIN, NOISE_MAX) / D_MAX
        noisy = clean + np.random.randn(*clean.shape).astype(np.float32) * sigma

        return (torch.from_numpy(noisy).unsqueeze(0),
                torch.from_numpy(clean).unsqueeze(0))


def main():
    print(f"Using device: {DEVICE}")

    all_paths = sorted(glob.glob(f"{VKITTI_DEPTH_ROOT}/*/*/frames/depth/Camera_0/*.png"))
    print(f"Found {len(all_paths)} depth images")

    random.shuffle(all_paths)
    n_train     = int(len(all_paths) * TRAIN_FRAC)
    train_paths = all_paths[:n_train]
    val_paths   = all_paths[n_train:]

    train_ds = DispDataset(train_paths)
    val_ds   = DispDataset(val_paths, patches_per_img=4)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = DnCNN(depth=8, channels=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

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

        print(f"Epoch {epoch:3d}/{EPOCHS} | train RMSE {np.sqrt(train_loss) * D_MAX:.3f}px"
              f" | val RMSE {np.sqrt(val_loss) * D_MAX:.3f}px")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  -> saved {OUTPUT_PATH}")

    print(f"Done. Best val RMSE: {np.sqrt(best_val) * D_MAX:.3f} px")


if __name__ == "__main__":
    main()
