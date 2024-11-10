import os.path as osp

import cv2
import numpy as np


def histogram_equalization(img):
    """Returns the image after histogram equalization.
    Args:
        img: the input image to be executed for histogram equalization.
    Returns:
        res_img: the output image after histogram equalization.
    """
    # TODO: implement the histogram equalization function.
    # Placeholder that you can delete. An image with all zeros.
    res_img = np.zeros_like(img)

    return res_img


def local_histogram_equalization(img):
    """Returns the image after local histogram equalization.
    Args:
        img: the input image to be executed for local histogram equalization.
    Returns:
        res_img: the output image after local histogram equalization.
    """
    # TODO: implement the local histogram equalization function.
    # Placeholder that you can delete. An image with all zeros.
    res_img = np.zeros_like(img)

    return res_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "moon.png"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res_hist_equalization = histogram_equalization(img)
    res_local_hist_equalization = local_histogram_equalization(img)

    cv2.imwrite(osp.join(root_dir, "HistEqualization.jpg"), res_hist_equalization)
    cv2.imwrite(
        osp.join(root_dir, "LocalHistEqualization.jpg"), res_local_hist_equalization
    )
