# Stereo Depth Estimation with Learned Regularizers

**Authors:** Joana Mizrahi, Jonathan Dumanski
**Course:** EE367 Computational Imaging, Stanford University, Winter 2026
**Contact:** jmizrahi@stanford.edu, jdumansk@stanford.edu

---

## Overview

We frame stereo matching as an inverse problem and compare regularizers:

| Method | Regularizer | EPE (px) | bad3 (%) |
|---|---|---|---|
| WTA | none (argmin) | 20.19 | 44.2 |
| TV-ADMM | total variation | 14.58 | 41.9 |
| HQS (Gaussian denoiser) | DnCNN (mismatched) | 20.93 | 51.0 |
| SGM | dynamic programming | 7.63 | 22.6 |
| **HQS-WTA-L1 (ours)** | **DnCNN (domain-matched)** | **6.79** | **23.2** |
| **SGM+denoiser (ours)** | **DnCNN post-processing** | **4.19** | **21.0** |

Averaged over 20 KITTI 2015 frames. Denoisers trained on vKITTI 2.0 (synthetic), evaluated on real KITTI data.

---

## Dependencies

Standard Python scientific stack — no special installations needed:

```
numpy scipy opencv-python torch matplotlib
```

Install with:
```bash
pip install numpy scipy opencv-python torch matplotlib
```

---

## Dataset Setup

The datasets are not included due to size. Download and place them as follows:

**KITTI 2015** (stereo evaluation):
Download `data_scene_flow.zip` from https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php
Extract so the structure is:
```
data_scene_flow/training/image_2/
data_scene_flow/training/image_3/
data_scene_flow/training/disp_occ_0/
```

**vKITTI 2.0** (denoiser training):
Download RGB and depth archives from https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
Extract so the structure is:
```
vkitti_2.0.3_rgb/Scene01/clone/frames/rgb/Camera_0/*.jpg
vkitti_2.0.3_depth/Scene01/clone/frames/depth/Camera_0/*.png
```
Scenes: 01, 02, 06, 18, 20. Scene20 is held out for testing.

---

## File Structure

```
stereo_matching/
├── main.py                  Entry point: evaluation and figure generation
├── costs.py                 Census transform cost volume computation
├── recover.py               All disparity recovery methods (SGM, TV-ADMM, HQS)
├── model.py                 DnCNN architecture
├── utils.py                 TV utilities (grad, div, Chambolle's algorithm)
├── evaluate.py              KITTI evaluation metrics (EPE, bad3)
├── models/
│   ├── denoiser_wta_l1.pth  WTA-trained DnCNN (domain-matched, main method)
│   ├── denoiser_sgm.pth     SGM-trained DnCNN (post-processing refinement)
│   └── denoiser_gaussian.pth Gaussian-noise DnCNN (baseline comparison)
└── training/
    ├── train_gaussian_denoiser.py Train the Gaussian-noise DnCNN (baseline)
    ├── train_wta_denoiser.py      Train the WTA domain-matched DnCNN (main method)
    └── train_sgm_denoiser.py      Train the SGM post-processing DnCNN
```

---

## Running the Code

**Evaluate all methods on KITTI 000000_10:**
```bash
python main.py
```

**Evaluate on vKITTI Scene20** (held-out test set):
```bash
python main.py --vkitti
```

**Benchmark over 10 KITTI frames:**
```bash
python main.py --benchmark
```

**Generate all report figures:**
```bash
python main.py --figures
```
Figures are saved to `poster/Figures/`.

---

## Retraining the Denoisers

To retrain from scratch (requires vKITTI dataset and a GPU):

```bash
# Train Gaussian denoiser (baseline -- domain mismatched)
python training/train_gaussian_denoiser.py

# Train WTA domain-matched denoiser (main method)
python training/train_wta_denoiser.py

# Train SGM post-processing denoiser
python training/train_sgm_denoiser.py
```

Checkpoints are saved to `models/`.

---

## Method Summary

**Cost volume:** Census transform with 7x7 window, Hamming distance, aggregated with 11x11 sum-pooling. 192 disparity levels.

**TV-ADMM:** ADMM with z=d splitting. d-update solved per-pixel via argmin over cost volume. z-update uses Chambolle's algorithm (proximal of TV).

**HQS (Half-Quadratic Splitting):** Removes the dual variable u from ADMM. Denoiser always sees d directly, preventing out-of-distribution drift. d-update: argmin over cost volume. z-update: DnCNN forward pass.

**Domain-matched denoiser:** DnCNN (8 layers, 64 channels) trained on (WTA output -> GT disparity) pairs from vKITTI using L1 loss. This matches the inference input distribution exactly, unlike a Gaussian-noise denoiser.