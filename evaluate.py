import numpy as np
import cv2

# load_gt_disp("data_scene_flow/training/disp_occ_0/000000_10.png")
# -> disp: (H, W) float32, real disparity values
# -> mask: (H, W) bool, True where GT is valid (pixel != 0)
def load_gt_disp(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    mask = raw > 0
    disp = raw / 256.0
    return disp, mask

# end point error
def epe(pred, gt, mask):
    return np.abs(pred[mask] - gt[mask]).mean()

def bad_px(pred, gt, mask, threshold=3.0):
    err = np.abs(pred[mask] - gt[mask])
    return (err > threshold).mean() * 100

def evaluate(pred_disp, gt_path):
    gt, mask = load_gt_disp(gt_path)
    return {
        "epe":    epe(pred_disp, gt, mask),
        "bad3":   bad_px(pred_disp, gt, mask, threshold=3.0),
    }


if __name__ == "__main__":
    # Sanity check: load one GT and print stats
    path = "data_scene_flow/training/disp_occ_0/000000_10.png"
    disp, mask = load_gt_disp(path)
    print("shape:       ", disp.shape)
    print("valid pixels:", mask.sum(), "/", mask.size, f"({100*mask.mean():.1f}%)")
    print("disp range:  ", f"{disp[mask].min():.2f} - {disp[mask].max():.2f}")

    import matplotlib.pyplot as plt
    plt.imshow(disp, cmap="plasma", vmin=0, vmax=disp[mask].max())
    plt.colorbar(label="disparity (px)")
    plt.title("GT disparity 000000_10")
    plt.show()
