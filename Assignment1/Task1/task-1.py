import numpy as np
import cv2


def image_t(im, scale=1.0, rot=45, trans=(50, -50)):
    # TODO Write "images affine transformation" function based on the illustration in specification.
    # 1. Transform the given angle θ to radian (π ∗θ/180)
    radian = np.deg2rad(rot)  # θ * π / 180

    # 2.  Sample three points(triangle) in the given image, and compute corresponding point coordinates in the target.
    # get height, weight, center
    print("img shape: ", im.shape)
    height, width, _ = im.shape
    center = np.float32([width / 2, height / 2])
    # choose three points - three corners
    p1 = [0, 0]
    p2 = [width - 1, 0]
    p3 = [0, height - 1]
    src_triangle = np.float32([p1, p2, p3])
    # rotate by the center
    src_triangle_center = src_triangle - center
    # generate the scale+rotation matrix
    matrix = np.array([[scale * np.cos(radian), -scale * np.sin(radian)],
                       [scale * np.sin(radian), scale * np.cos(radian)]])
    # get the dst_triangle
    dst_triangle = np.float32(np.dot(src_triangle_center, matrix) + np.array(trans)) + center

    # calculate transformation matrix using these six points
    M = cv2.getAffineTransform(src_triangle, dst_triangle)

    # obtain the target image
    resultImg = cv2.warpAffine(src=im, M=M, dsize=(height, width))

    return resultImg


if __name__ == '__main__':
    im = cv2.imread('../images/pearl.jpeg')

    scale = 0.5
    rot = 45
    trans = (0, 0)
    result = image_t(im, scale, rot, trans)
    cv2.imwrite('../results/affine_result.png', result)
    cv2.imshow('result', result)
    cv2.waitKey(0)
