import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp


def show_img(path):
    """ 读取并展示图片

    :param path: 图片路径
    :return:
    """
    img = mp.imread(path)
    print('图片的shape:', img.shape)
    plt.imshow(img)
    plt.show()


def image_normalization(img, img_min=0, img_max=255):
    """数据正则化,将数据从一个小范围变换到另一个范围
        默认参数：从（0,1） -> (0,255)
    :param img: 输入数据
    :param img_min: 数据最小值
    :param img_max: 数据最大值
    :return: 返回变换后的结果结果
    """
    img = np.float32(img)
    epsilon = 1e-12
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min

    return img


def show_img_2(path):
    """ 利用opencv 读取并显示单通道图片

    :param path: 图片路径
    :return:
    """
    segmentation_img = cv2.imread(path, 0)
    # 创建一个与segmentation_image相同大小的全零矩阵，用于存储二值化后的mask
    mask = np.zeros(segmentation_img.shape, dtype=np.uint8)
    # 将分割图像中大于0的像素值设为前景（1），其余为背景（0）
    mask[segmentation_img > 0] = 1
    # 显示生成的mask
    cv2.imshow('Mask', mask * 255)  # 将mask乘以255以便于显示（将0-1映射到0-255）


def show_img_3(path):
    """读取分割图像并生成0,1,2,3四个label的掩码图像
    :param path: 图像路径
    """
    # 读取图像
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite("results/ori.png",img)

    # 创建与输入图像相同大小的掩码
    mask = np.zeros(img.shape, dtype=np.uint8)

    # 将不同范围的像素值设置为不同的标签
    mask[img == 0] = 0  # 标签 0
    mask[img == 1] = 1  # 标签 1
    mask[img == 2] = 2  # 标签 2
    mask[img == 3] = 3  # 标签 3

    # 将掩码中的标签放大显示（不同标签对应不同灰度值）
    mask_display = mask * 85  # 将标签值映射到 0, 85, 170, 255
    cv2.imshow('Multi-label Mask', mask_display)
    cv2.imwrite("results/mask.png",mask_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = 'data/matting_human_sample/matting/1803151818/matting_00000000/1803151818-00000009.png'
mask_path = 'data/EG1800/Labels/00001.png'
mask_path1 = 'data/new_archive/annotations/train/0a2f0ae1-65df-401a-a547-51b95eeb59cf.png'

show_img_3(mask_path1)

cv2.waitKey(0)
