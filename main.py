"""
Main evaluation script for stereo disparity estimation.

Reproduces all results from the report on KITTI 2015 and vKITTI 2.0.

Quick start:
    python main.py                     # runs test_kitti_all() by default
    python main.py --vkitti            # evaluate on vKITTI Scene20
    python main.py --benchmark         # avg over 10 KITTI frames (slow)
    python main.py --figures           # generate all report figures

Dataset paths (relative to this file):
    KITTI 2015 : data_scene_flow/training/{image_2,image_3,disp_occ_0}/
    vKITTI 2.0 : vkitti_2.0.3_rgb/  and  vkitti_2.0.3_depth/
"""

import argparse
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from costs import init_cost_census
from evaluate import load_gt_disp, evaluate
from recover import (
    recover_disp_sgm,
    recover_disp_tv_v2,
    recover_disp_learned,
    recover_disp_hqs,
    recover_disp_hqs_wta_l1,
    recover_disp_sgm_denoised,
)

# ---- Shared constants ----
D_MAX    = 192
WIN_SIZE = 11

# Default KITTI test image
KITTI_L  = "data_scene_flow/training/image_2/000000_10.png"
KITTI_R  = "data_scene_flow/training/image_3/000000_10.png"
KITTI_GT = "data_scene_flow/training/disp_occ_0/000000_10.png"

# vKITTI camera calibration: fB = focal_length * baseline (px * m)
F_B = 725.0087 * 0.532725


# ---- vKITTI helpers ----

def load_vkitti_gt(depth_path):
    """Load vKITTI depth PNG (uint16 in cm) -> disparity map."""
    raw   = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid = raw < 65535  # 65535 = no return
    disp  = np.zeros_like(raw)
    disp[valid] = np.clip(F_B / (raw[valid] / 100.0), 0, D_MAX)
    return disp, valid


def evaluate_vkitti(pred_disp, depth_path):
    """Evaluate predicted disparity against vKITTI GT (MAE and bad3)."""
    gt, mask = load_vkitti_gt(depth_path)
    epe  = np.abs(pred_disp[mask] - gt[mask]).mean()
    bad3 = (np.abs(pred_disp[mask] - gt[mask]) > 3.0).mean() * 100
    return {"epe": epe, "bad3": bad3}


def _vkitti_paths(scene="Scene20", frame_idx=None):
    """Return (left, right, depth) paths for a vKITTI frame."""
    left_paths = sorted(glob.glob(
        f"vkitti_2.0.3_rgb/*/clone/frames/rgb/Camera_0/*.jpg"))
    if scene:
        left_paths = [p for p in left_paths if f"/{scene}/" in p]
    lp = left_paths[frame_idx] if frame_idx is not None else random.choice(left_paths)
    rp = lp.replace("Camera_0", "Camera_1")
    dp = (lp.replace("vkitti_2.0.3_rgb", "vkitti_2.0.3_depth")
            .replace("rgb/Camera_0/rgb_", "depth/Camera_0/depth_")
            .replace(".jpg", ".png"))
    return lp, rp, dp


# ---- Evaluation functions ----

def test_kitti_all(mu_hqs=0.01, mu_tv=0.05, lam_tv=0.1, mu_pnp=0.05):
    """Run all methods on KITTI 000000_10 and print a results table."""
    L_img = cv2.imread(KITTI_L)
    R_img = cv2.imread(KITTI_R)

    print(f"{'Method':<22} {'EPE (px)':>9} {'bad3 (%)':>9}")
    print("-" * 44)

    # WTA (no regularizer)
    L_g = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    R_g = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    cost = init_cost_census(L_g, R_g, D_MAX, WIN_SIZE, census_win=7)
    wta  = np.argmin(cost, axis=2).astype(np.float32)
    m    = evaluate(wta, KITTI_GT)
    print(f"{'WTA':<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")

    # SGM
    sgm = recover_disp_sgm(L_img, R_img, D_MAX, WIN_SIZE)
    m   = evaluate(sgm, KITTI_GT)
    print(f"{'SGM':<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")

    # TV-ADMM
    tv = recover_disp_tv_v2(L_img, R_img, mu_tv, lam_tv, D_MAX, WIN_SIZE)
    m  = evaluate(tv, KITTI_GT)
    print(f"{'TV-ADMM':<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")

    # PnP-ADMM (Gaussian denoiser)
    pnp = recover_disp_learned(L_img, R_img, mu_pnp, D_MAX, WIN_SIZE)
    m   = evaluate(pnp, KITTI_GT)
    print(f"{'PnP-ADMM (Gaussian)':<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")

    # HQS (Gaussian denoiser)
    hqs = recover_disp_hqs(L_img, R_img, mu_hqs, D_MAX, WIN_SIZE)
    m   = evaluate(hqs, KITTI_GT)
    print(f"{'HQS (Gaussian)':<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")

    # HQS-WTA-L1 (main method)
    hqs_wta = recover_disp_hqs_wta_l1(L_img, R_img, mu_hqs, D_MAX, WIN_SIZE)
    m_hqs   = evaluate(hqs_wta, KITTI_GT)
    print(f"{'HQS-WTA-L1 (ours)':<22} {m_hqs['epe']:>9.3f} {m_hqs['bad3']:>8.1f}%")

    # SGM + DnCNN refinement
    sgm_dn = recover_disp_sgm_denoised(L_img, R_img, D_MAX, WIN_SIZE)
    m      = evaluate(sgm_dn, KITTI_GT)
    print(f"{'SGM+denoiser':<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")


def test_on_vkitti(mu_hqs=0.01, mu_tv=0.05, lam_tv=0.1, mu_pnp=0.05,
                   scene="Scene20", frame_idx=None):
    """Run selected methods on a single vKITTI frame (Scene20 = held-out test set)."""
    lp, rp, dp = _vkitti_paths(scene, frame_idx)
    print(f"Frame: {lp}\n")

    L = cv2.imread(lp)
    R = cv2.imread(rp)

    print(f"{'Method':<22} {'EPE (px)':>9} {'bad3 (%)':>9}")
    print("-" * 44)

    L_g = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    R_g = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    cost = init_cost_census(L_g, R_g, D_MAX, WIN_SIZE, census_win=7)
    wta  = np.argmin(cost, axis=2).astype(np.float32)
    m    = evaluate_vkitti(wta, dp)
    print(f"{'WTA':<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")

    for name, fn, args in [
        ("SGM",              recover_disp_sgm,        (D_MAX, WIN_SIZE)),
        ("TV-ADMM",          recover_disp_tv_v2,       (mu_tv, lam_tv, D_MAX, WIN_SIZE)),
        ("PnP-ADMM (Gauss)", recover_disp_learned,    (mu_pnp, D_MAX, WIN_SIZE)),
        ("HQS (Gaussian)",   recover_disp_hqs,         (mu_hqs, D_MAX, WIN_SIZE)),
        ("HQS-WTA-L1 (ours)",recover_disp_hqs_wta_l1, (mu_hqs, D_MAX, WIN_SIZE)),
        ("SGM+denoiser",     recover_disp_sgm_denoised,(D_MAX, WIN_SIZE)),
    ]:
        disp = fn(L, R, *args)
        m    = evaluate_vkitti(disp, dp)
        print(f"{name:<22} {m['epe']:>9.3f} {m['bad3']:>8.1f}%")


def benchmark_kitti(n=10, mu_hqs=0.01, mu_tv=0.05, lam_tv=0.1, mu_pnp=0.05):
    """Average EPE and bad3 over N random KITTI training frames for all methods."""
    left_imgs = sorted(glob.glob("data_scene_flow/training/image_2/*_10.png"))
    random.seed(42)
    selected = random.sample(left_imgs, min(n, len(left_imgs)))

    methods = ["WTA", "SGM", "TV-ADMM", "PnP-ADMM", "HQS", "HQS-WTA-L1", "SGM+denoiser"]
    totals  = {m: {"epe": 0.0, "bad3": 0.0, "count": 0} for m in methods}

    for i, lp in enumerate(selected):
        rp = lp.replace("image_2", "image_3")
        gp = lp.replace("image_2", "disp_occ_0")
        L  = cv2.imread(lp)
        R  = cv2.imread(rp)
        if L is None or R is None:
            continue
        print(f"[{i+1}/{len(selected)}] {os.path.basename(lp)}")

        L_g = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        R_g = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        cost = init_cost_census(L_g, R_g, D_MAX, WIN_SIZE, census_win=7)
        wta  = np.argmin(cost, axis=2).astype(np.float32)

        runs = [
            ("WTA",          wta, None),
            ("SGM",          recover_disp_sgm,         (L, R, D_MAX, WIN_SIZE)),
            ("TV-ADMM",      recover_disp_tv_v2,        (L, R, mu_tv, lam_tv, D_MAX, WIN_SIZE)),
            ("PnP-ADMM",     recover_disp_learned,      (L, R, mu_pnp, D_MAX, WIN_SIZE)),
            ("HQS",          recover_disp_hqs,           (L, R, mu_hqs, D_MAX, WIN_SIZE)),
            ("HQS-WTA-L1",   recover_disp_hqs_wta_l1,   (L, R, mu_hqs, D_MAX, WIN_SIZE)),
            ("SGM+denoiser", recover_disp_sgm_denoised,  (L, R, D_MAX, WIN_SIZE)),
        ]
        for name, fn, args in runs:
            try:
                disp = fn if args is None else fn(*args)
                m    = evaluate(disp, gp)
                totals[name]["epe"]   += m["epe"]
                totals[name]["bad3"]  += m["bad3"]
                totals[name]["count"] += 1
            except Exception:
                pass

    print(f"\n{'Method':<22} {'EPE (px)':>9} {'bad3 (%)':>9}  (avg over {n} frames)")
    print("-" * 46)
    for name in methods:
        c = totals[name]["count"]
        if c > 0:
            print(f"{name:<22} {totals[name]['epe']/c:>9.3f} {totals[name]['bad3']/c:>8.1f}%")


# ---- Figure generation (for report) ----

def save_domain_mismatch_figure():
    """3-panel figure: Gaussian denoiser vs WTA-trained denoiser vs GT (KITTI)."""
    L_img = cv2.imread(KITTI_L)
    R_img = cv2.imread(KITTI_R)
    gt, _ = load_gt_disp(KITTI_GT)

    gauss  = recover_disp_learned(L_img, R_img,    mu=0.05, D_max=D_MAX, win_size=WIN_SIZE)
    wta_l1 = recover_disp_hqs_wta_l1(L_img, R_img, mu=0.001, D_max=D_MAX, win_size=WIN_SIZE)
    m_g    = evaluate(gauss,  KITTI_GT)
    m_w    = evaluate(wta_l1, KITTI_GT)

    methods = [
        (gauss,  f"HQS (Gaussian denoiser)\nEPE={m_g['epe']:.2f}  bad3={m_g['bad3']:.1f}%"),
        (wta_l1, f"HQS-WTA-L1 (domain-matched)\nEPE={m_w['epe']:.2f}  bad3={m_w['bad3']:.1f}%"),
        (gt,     "Ground Truth"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, (img, title) in zip(axes, methods):
        ax.imshow(img, cmap="plasma", vmin=0, vmax=D_MAX)
        ax.set_title(title, fontsize=9, pad=3)
        ax.axis("off")
    plt.subplots_adjust(wspace=0.02, top=0.88, bottom=0.01, left=0.01, right=0.99)
    plt.suptitle("Domain mismatch: Gaussian vs. WTA-trained denoiser (KITTI 000000_10)", fontsize=10)
    os.makedirs("poster/Figures", exist_ok=True)
    plt.savefig("poster/Figures/domain_mismatch.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved poster/Figures/domain_mismatch.png")


def save_mu_ablation_plot(mus=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1), n=5):
    """Sweep mu for HQS-WTA-L1 over n KITTI frames and save EPE + bad3 dual-axis plot."""
    left_imgs = sorted(glob.glob("data_scene_flow/training/image_2/*_10.png"))
    random.seed(42)
    selected = random.sample(left_imgs, min(n, len(left_imgs)))

    epe_vals, bad3_vals = [], []
    for mu in mus:
        epe_sum, bad3_sum, count = 0.0, 0.0, 0
        for lp in selected:
            rp = lp.replace("image_2", "image_3")
            gp = lp.replace("image_2", "disp_occ_0")
            L  = cv2.imread(lp)
            R  = cv2.imread(rp)
            if L is None or R is None:
                continue
            try:
                d = recover_disp_hqs_wta_l1(L, R, mu, D_MAX, WIN_SIZE)
                m = evaluate(d, gp)
                epe_sum  += m["epe"]
                bad3_sum += m["bad3"]
                count    += 1
            except Exception:
                pass
        epe_vals.append(epe_sum / count if count else float("nan"))
        bad3_vals.append(bad3_sum / count if count else float("nan"))
        print(f"mu={mu:<7} EPE={epe_vals[-1]:.3f}  bad3={bad3_vals[-1]:.1f}%")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = list(range(len(mus)))
    ax1.plot(x, epe_vals,  "o-",  color="steelblue", label="EPE (px)")
    ax1.set_ylabel("EPE (px)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(x, bad3_vals, "s--", color="tomato",    label="bad3 (%)")
    ax2.set_ylabel("bad3 (%)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(m) for m in mus], rotation=30, ha="right", fontsize=8)
    ax1.set_xlabel(r"$\mu$")
    ax1.set_title(r"HQS-WTA-L1: EPE and bad3 vs. $\mu$ (avg over 5 KITTI frames)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs("poster/Figures", exist_ok=True)
    plt.savefig("poster/Figures/mu_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved poster/Figures/mu_ablation.png")


def save_training_patches_figure(n_patches=6, patch_size=64, seed=42):
    """Show example (WTA input, GT target) training patches from vKITTI."""
    np.random.seed(seed)
    random.seed(seed)

    left_imgs = sorted(
        glob.glob("vkitti_2.0.3_rgb/Scene01/clone/frames/rgb/Camera_0/*.jpg"))[:20]
    patches_wta, patches_gt = [], []

    for lp in random.sample(left_imgs, min(10, len(left_imgs))):
        rp = lp.replace("Camera_0", "Camera_1")
        dp = (lp.replace("vkitti_2.0.3_rgb", "vkitti_2.0.3_depth")
                .replace("rgb/Camera_0", "depth/Camera_0")
                .replace("rgb_", "depth_").replace(".jpg", ".png"))
        L = cv2.imread(lp)
        R = cv2.imread(rp)
        D = cv2.imread(dp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if L is None or R is None or D is None:
            continue

        L_g = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        R_g = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        cost = init_cost_census(L_g, R_g, D_MAX, WIN_SIZE, census_win=7)
        wta  = np.argmin(cost, axis=2).astype(np.float32)

        depth = D.astype(np.float32) / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            gt_disp = np.where(depth > 0, F_B / depth, 0).astype(np.float32)

        h, w = wta.shape
        for _ in range(20):
            y  = np.random.randint(0, h - patch_size)
            x  = np.random.randint(0, w - patch_size)
            gp = gt_disp[y:y+patch_size, x:x+patch_size]
            if np.mean(gp > 0) < 0.7:
                continue
            patches_wta.append(wta[y:y+patch_size, x:x+patch_size])
            patches_gt.append(gp)
            if len(patches_wta) >= n_patches:
                break
        if len(patches_wta) >= n_patches:
            break

    vmax = np.percentile([p for p in patches_gt if p.max() > 0], 95)
    fig, axes = plt.subplots(2, n_patches, figsize=(n_patches * 2.5, 5))
    for i in range(n_patches):
        axes[0, i].imshow(patches_wta[i], cmap="plasma", vmin=0, vmax=vmax)
        axes[0, i].axis("off")
        axes[1, i].imshow(patches_gt[i],  cmap="plasma", vmin=0, vmax=vmax)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("WTA", fontsize=9)
    axes[1, 0].set_ylabel("GT",  fontsize=9)
    plt.subplots_adjust(wspace=0.02, hspace=0.05, top=0.93, bottom=0.01, left=0.06, right=0.99)
    plt.suptitle("Example training patches: WTA disparity (top) vs. ground truth (bottom)", fontsize=10)
    os.makedirs("poster/Figures", exist_ok=True)
    plt.savefig("poster/Figures/training_patches.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved poster/Figures/training_patches.png")


def save_dataset_figure():
    """Side-by-side KITTI and vKITTI stereo pairs + GT disparity maps."""
    kitti_l  = cv2.cvtColor(cv2.imread(KITTI_L), cv2.COLOR_BGR2RGB)
    kitti_r  = cv2.cvtColor(cv2.imread(KITTI_R), cv2.COLOR_BGR2RGB)
    kitti_gt, _ = load_gt_disp(KITTI_GT)

    vkitti_l = cv2.cvtColor(
        cv2.imread("vkitti_2.0.3_rgb/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"),
        cv2.COLOR_BGR2RGB)
    vkitti_r = cv2.cvtColor(
        cv2.imread("vkitti_2.0.3_rgb/Scene01/clone/frames/rgb/Camera_1/rgb_00000.jpg"),
        cv2.COLOR_BGR2RGB)
    vkitti_d = cv2.imread(
        "vkitti_2.0.3_depth/Scene01/clone/frames/depth/Camera_0/depth_00000.png",
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    vkitti_gt = np.where(vkitti_d > 0, F_B / (vkitti_d.astype(np.float32) / 100.0), 0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    axes[0, 0].imshow(kitti_l);  axes[0, 0].set_title("KITTI 2015 - Left",  fontsize=9)
    axes[0, 1].imshow(kitti_r);  axes[0, 1].set_title("KITTI 2015 - Right", fontsize=9)
    axes[0, 2].imshow(kitti_gt,  cmap="plasma", vmin=0, vmax=D_MAX)
    axes[0, 2].set_title("KITTI - GT disparity (LiDAR, sparse)", fontsize=9)
    axes[1, 0].imshow(vkitti_l); axes[1, 0].set_title("vKITTI 2.0 - Left",  fontsize=9)
    axes[1, 1].imshow(vkitti_r); axes[1, 1].set_title("vKITTI 2.0 - Right", fontsize=9)
    axes[1, 2].imshow(vkitti_gt, cmap="plasma", vmin=0, vmax=D_MAX)
    axes[1, 2].set_title("vKITTI - GT disparity (synthetic, dense)", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    plt.subplots_adjust(wspace=0.02, hspace=0.08, top=0.93, bottom=0.01, left=0.01, right=0.99)
    plt.suptitle("Evaluation: KITTI 2015 (real) vs. Training: vKITTI 2.0 (synthetic)", fontsize=10)
    os.makedirs("poster/Figures", exist_ok=True)
    plt.savefig("poster/Figures/datasets.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved poster/Figures/datasets.png")


def save_all_figures():
    """Generate all figures used in the report."""
    print("Generating dataset figure...")
    save_dataset_figure()
    print("Generating training patches figure...")
    save_training_patches_figure()
    print("Generating domain mismatch figure...")
    save_domain_mismatch_figure()
    print("Generating mu ablation plot (slowest ~5 min)...")
    save_mu_ablation_plot()
    print("Done.")


# ---- Entry point ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo matching evaluation")
    parser.add_argument("--vkitti",    action="store_true", help="Evaluate on vKITTI Scene20")
    parser.add_argument("--benchmark", action="store_true", help="Avg over 10 KITTI frames")
    parser.add_argument("--figures",   action="store_true", help="Generate all report figures")
    args = parser.parse_args()

    if args.vkitti:
        test_on_vkitti()
    elif args.benchmark:
        benchmark_kitti()
    elif args.figures:
        save_all_figures()
    else:
        test_kitti_all()
