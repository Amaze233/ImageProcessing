import numpy as np
import cv2


def image_t(im, scale=1.0, rot=45, trans=(50, -50)):
    # TODO Write "images affine transformation" function based on the illustration in specification.
    # Return transformed result images
    cv2.warpAffine()
    result = None
    return result


if __name__ == '__main__':
    im = cv2.imread('../images/pearl.jpeg')

    scale = 0.5
    rot = 45
    trans = (50, -50)
    result = image_t(im, scale, rot, trans)
    cv2.imwrite('./results/affine_result.png', result)
