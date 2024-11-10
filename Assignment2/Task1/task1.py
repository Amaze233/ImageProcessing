import math
import os.path as osp

import cv2
import numpy as np


def gaussian_filter(img, kernel_size, sigma):
    """Returns the image after Gaussian filter.
    Args:
        img: the input image to be Gaussian filtered.
        kernel_size: the kernel size in both the X and Y directions.
        sigma: the standard deviation in both the X and Y directions.
    Returns:
        res_img: the output image after Gaussian filter.
    """
    # TODO: implement the Gaussian filter function.
    # Placeholder that you can delete. An image with all zeros.
    res_img = np.zeros_like(img)
    width, height, _ = img.shape

    # generate the gaussian kernel matrix (kernel_size x kernel_size)
    gaussian_kernel = np.zeros((kernel_size, kernel_size), np.float32)
    k = kernel_size // 2  # (k,k) the center of the kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            # calculate the norm of (i,j) to center
            norm = math.pow(i - k, 2) + math.pow(j - k, 2)
            # calculate the gaussian kernel
            gaussian_kernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2))) / math.sqrt(2 * math.pi * sigma * sigma)

    # normalize the kernel
    gaussian_kernel /= np.sum(gaussian_kernel)

    # apply the gaussian kernel to the image
    # don't process the edge values
    for i in range(width - 2*k):
        for j in range(height - 2*k):
            for c in range(3):
                ori_matrix = img[i:i + kernel_size, j:j + kernel_size, c]
                res_img[i+k, j+k, c] = np.sum(ori_matrix * gaussian_kernel)

    return res_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "images/Lena-RGB.jpg"))
    kernel_size = 5
    sigma = 1
    res_img = gaussian_filter(img, kernel_size, sigma)
    cv2.imshow("result", res_img)
    cv2.waitKey(0)
    cv2.imwrite(osp.join(root_dir, "results/gaussian_result.jpg"), res_img)
