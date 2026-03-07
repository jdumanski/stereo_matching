import cv2
import matplotlib.pyplot as plt

from recover import recover_disp_tv

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
