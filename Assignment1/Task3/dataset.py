# datasets.py

# Define custom dataset classes and data loading functions.
# This file should include:
# 1. Custom Dataset class(es) inheriting from torch.utils.data.Dataset
# 2. Data loading and preprocessing functions
# 3. Data augmentation techniques (if applicable)
# 4. Functions to split data into train/val/test sets
# 5. Any necessary data transformations


# 1. Data Augmentation copied from data_aug.py
import os
import cv2
import math
import random
import scipy
import json
import copy
import base64
import zlib
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance, ImageOps, ImageFile

import sys
# sys.path.insert(0, '/home/dongx12/Data/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# global parameter
set_ratio = 0.5


def load_json(fileName):
    with open(fileName, 'r') as data_file:
        anno = json.load(data_file)
    return anno


def mask_to_bbox(mask):
    site = np.where(mask > 0)
    bbox = [np.min(site[1]), np.min(site[0]), np.max(site[1]), np.max(site[0])]
    return bbox


# ===================== generate edge for input image =====================
def show_edge(mask_ori):
    mask = mask_ori.copy()
    # find countours: img must be binary
    myImg = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    ret, binary = cv2.threshold(np.uint8(mask) * 255, 127, 255, cv2.THRESH_BINARY)
    countours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
    '''
    cv2.drawContours(myImg, countours, -1, 1, 10)
    diff = mask + myImg
    diff[diff < 2] = 0
    diff[diff == 2] = 1
    return diff
    '''
    cv2.drawContours(myImg, countours, -1, 1, 4)
    return myImg


# ===================== load mask =====================
def annToRLE(anno, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = anno['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = anno['segmentation']
    return rle


def annToMask(anno, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(anno, height, width)
    mask = maskUtils.decode(rle)
    return mask


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


# ===================== deformable data augmentation for input image =====================
def flip_data(width, keypoint_ori):
    keypoint = copy.deepcopy(keypoint_ori)
    for i in xrange(len(keypoint) / 3):
        keypoint[3 * i] = width - 1 - keypoint[3 * i]
    right = [2, 4, 6, 8, 10, 12, 14, 16]
    left = [1, 3, 5, 7, 9, 11, 13, 15]

    for i in xrange(len(left)):
        temp = copy.deepcopy(keypoint[3 * right[i]:3 * (right[i] + 1)])
        keypoint[3 * right[i]:3 * (right[i] + 1)] = keypoint[3 * left[i]:3 * (left[i] + 1)]
        keypoint[3 * left[i]:3 * (left[i] + 1)] = temp
    return keypoint


def data_aug_flip(image, mask):
    if random.random() < set_ratio:
        return image, mask, False
    return image[:, ::-1, :], mask[:, ::-1], True


def aug_matrix(img_w, img_h, bbox, w, h, angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=40):
    '''
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    '''
    ratio = 1.0 * (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (img_w * img_h)
    x_offset = (random.random() - 0.5) * 2 * offset
    y_offset = (random.random() - 0.5) * 2 * offset
    dx = (w - (bbox[2] + bbox[0])) / 2.0
    dy = (h - (bbox[3] + bbox[1])) / 2.0

    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0, 1.0]])

    angle = random.random() * (angle_range[1] - angle_range[0]) + angle_range[0]
    scale = random.random() * (scale_range[1] - scale_range[0]) + scale_range[0]
    scale *= np.mean([float(w) / (bbox[2] - bbox[0]), float(h) / (bbox[3] - bbox[1])])
    alpha = scale * math.cos(angle / 180.0 * math.pi)
    beta = scale * math.sin(angle / 180.0 * math.pi)

    centerx = w / 2.0 + x_offset
    centery = h / 2.0 + y_offset
    H = np.array([[alpha, beta, (1 - alpha) * centerx - beta * centery],
                  [-beta, alpha, beta * centerx + (1 - alpha) * centery],
                  [0, 0, 1.0]])

    H = H.dot(matrix_trans)[0:2, :]
    return H


# ===================== texture data augmentation for input image =====================
def data_aug_light(image):
    if random.random() < set_ratio:
        return image
    value = random.randint(-30, 30)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float32)
    hsv_image[:, :, 2] += value
    hsv_image[hsv_image > 255] = 255
    hsv_image[hsv_image < 0] = 0
    hsv_image = np.array(hsv_image, dtype=np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image


def data_aug_blur(image):
    if random.random() < set_ratio:
        return image

    select = random.random()
    if select < 0.3:
        kernalsize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernalsize, kernalsize), 0)
    elif select < 0.6:
        kernalsize = random.choice([3, 5])
        image = cv2.medianBlur(image, kernalsize)
    else:
        kernalsize = random.choice([3, 5])
        image = cv2.blur(image, (kernalsize, kernalsize))
    return image


def data_aug_color(image):
    if random.random() < set_ratio:
        return image
    random_factor = np.random.randint(4, 17) / 10.
    color_image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(4, 17) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(6, 15) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(8, 13) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)


def data_aug_noise(image):
    if random.random() < set_ratio:
        return image
    mu = 0
    sigma = random.random() * 10.0
    image = np.array(image, dtype=np.float32)
    image += np.random.normal(mu, sigma, image.shape)
    image[image > 255] = 255
    image[image < 0] = 0
    return image


# ===================== normalization for input image =====================
def padding(img_ori, mask_ori, size=224, padding_color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]

    img = np.zeros((max(height, width), max(height, width), 3)) + padding_color
    mask = np.zeros((max(height, width), max(height, width)))

    if (height > width):
        padding = int((height - width) / 2)
        img[:, padding:padding + width, :] = img_ori
        mask[:, padding:padding + width] = mask_ori
    else:
        padding = int((width - height) / 2)
        img[padding:padding + height, :, :] = img_ori
        mask[padding:padding + height, :] = mask_ori

    img = np.uint8(img)
    mask = np.uint8(mask)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)

    return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)


def Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32) / scale
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j, :, :, i] = (img[j, :, :, i] - mean[i]) * val[i]
        return img
    else:
        for i in range(len(mean)):
            img[:, :, i] = (img[:, :, i] - mean[i]) * val[i]
        return img


def Anti_Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j, :, :, i] = img[j, :, :, i] / val[i] + mean[i]
        return np.array(img * scale, np.uint8)
    else:
        for i in range(len(mean)):
            img[:, :, i] = img[:, :, i] / val[i] + mean[i]
        return np.array(img * scale, np.uint8)


# ===================== generate prior channel for input image =====================
def data_motion_blur(image, mask):
    if random.random() < set_ratio:
        return image, mask

    degree = random.randint(5, 30)
    angle = random.randint(0, 360)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    img_blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    mask_blurred = cv2.filter2D(mask, -1, motion_blur_kernel)

    cv2.normalize(img_blurred, img_blurred, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(mask_blurred, mask_blurred, 0, 1, cv2.NORM_MINMAX)
    return img_blurred, mask_blurred


def data_motion_blur_prior(prior):
    if random.random() < set_ratio:
        return prior

    degree = random.randint(5, 30)
    angle = random.randint(0, 360)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    prior_blurred = cv2.filter2D(prior, -1, motion_blur_kernel)
    return prior_blurred


def data_Affine(image, mask, height, width, ratio=0.05):
    if random.random() < set_ratio:
        return image, mask
    bias = np.random.randint(-int(height * ratio), int(width * ratio), 12)
    pts1 = np.float32([[0 + bias[0], 0 + bias[1]], [width + bias[2], 0 + bias[3]], [0 + bias[4], height + bias[5]]])
    pts2 = np.float32([[0 + bias[6], 0 + bias[7]], [width + bias[8], 0 + bias[9]], [0 + bias[10], height + bias[11]]])
    M = cv2.getAffineTransform(pts1, pts2)
    img_affine = cv2.warpAffine(image, M, (width, height))
    mask_affine = cv2.warpAffine(mask, M, (width, height))
    return img_affine, mask_affine


def data_Affine_prior(prior, height, width, ratio=0.05):
    if random.random() < set_ratio:
        return prior
    bias = np.random.randint(-int(height * ratio), int(width * ratio), 12)
    pts1 = np.float32([[0 + bias[0], 0 + bias[1]], [width + bias[2], 0 + bias[3]], [0 + bias[4], height + bias[5]]])
    pts2 = np.float32([[0 + bias[6], 0 + bias[7]], [width + bias[8], 0 + bias[9]], [0 + bias[10], height + bias[11]]])
    M = cv2.getAffineTransform(pts1, pts2)
    prior_affine = cv2.warpAffine(prior, M, (width, height))
    return prior_affine


def data_Perspective(image, mask, height, width, ratio=0.05):
    if random.random() < set_ratio:
        return image, mask
    bias = np.random.randint(-int(height * ratio), int(width * ratio), 16)
    pts1 = np.float32([[0 + bias[0], 0 + bias[1]], [height + bias[2], 0 + bias[3]],
                       [0 + bias[4], width + bias[5]], [height + bias[6], width + bias[7]]])
    pts2 = np.float32([[0 + bias[8], 0 + bias[9]], [height + bias[10], 0 + bias[11]],
                       [0 + bias[12], width + bias[13]], [height + bias[14], width + bias[15]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_perspective = cv2.warpPerspective(image, M, (width, height))
    mask_perspective = cv2.warpPerspective(mask, M, (width, height))
    return img_perspective, mask_perspective


def data_Perspective_prior(prior, height, width, ratio=0.05):
    if random.random() < set_ratio:
        return prior
    bias = np.random.randint(-int(height * ratio), int(width * ratio), 16)
    pts1 = np.float32([[0 + bias[0], 0 + bias[1]], [height + bias[2], 0 + bias[3]],
                       [0 + bias[4], width + bias[5]], [height + bias[6], width + bias[7]]])
    pts2 = np.float32([[0 + bias[8], 0 + bias[9]], [height + bias[10], 0 + bias[11]],
                       [0 + bias[12], width + bias[13]], [height + bias[14], width + bias[15]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    prior_perspective = cv2.warpPerspective(prior, M, (width, height))
    return prior_perspective


def data_ThinPlateSpline(image, mask, height, width, ratio=0.05):
    if random.random() < set_ratio:
        return image, mask
    bias = np.random.randint(-int(height * ratio), int(width * ratio), 16)
    tps = cv2.createThinPlateSplineShapeTransformer()
    sshape = np.array([[0 + bias[0], 0 + bias[1]], [height + bias[2], 0 + bias[3]],
                       [0 + bias[4], width + bias[5]], [height + bias[6], width + bias[7]]], np.float32)
    tshape = np.array([[0 + bias[8], 0 + bias[9]], [height + bias[10], 0 + bias[11]],
                       [0 + bias[12], width + bias[13]], [height + bias[14], width + bias[15]]], np.float32)
    sshape = sshape.reshape(1, -1, 2)
    tshape = tshape.reshape(1, -1, 2)
    matches = list()
    matches.append(cv2.DMatch(0, 0, 0))
    matches.append(cv2.DMatch(1, 1, 0))
    matches.append(cv2.DMatch(2, 2, 0))
    matches.append(cv2.DMatch(3, 3, 0))

    tps.estimateTransformation(tshape, sshape, matches)
    res = tps.warpImage(image)
    res_mask = tps.warpImage(mask)
    return res, res_mask


def data_ThinPlateSpline_prior(prior, height, width, ratio=0.05):
    if random.random() < set_ratio:
        return prior
    bias = np.random.randint(-int(height * ratio), int(width * ratio), 16)
    tps = cv2.createThinPlateSplineShapeTransformer()
    sshape = np.array([[0 + bias[0], 0 + bias[1]], [height + bias[2], 0 + bias[3]],
                       [0 + bias[4], width + bias[5]], [height + bias[6], width + bias[7]]], np.float32)
    tshape = np.array([[0 + bias[8], 0 + bias[9]], [height + bias[10], 0 + bias[11]],
                       [0 + bias[12], width + bias[13]], [height + bias[14], width + bias[15]]], np.float32)
    sshape = sshape.reshape(1, -1, 2)
    tshape = tshape.reshape(1, -1, 2)
    matches = list()
    matches.append(cv2.DMatch(0, 0, 0))
    matches.append(cv2.DMatch(1, 1, 0))
    matches.append(cv2.DMatch(2, 2, 0))
    matches.append(cv2.DMatch(3, 3, 0))

    tps.estimateTransformation(tshape, sshape, matches)
    prior = tps.warpImage(prior)
    return prior


# 2.Datasets Portrait Segmentation copied fromdatasets_portraitseg.py
import torch
import torch.utils.data as data

import os
import cv2
import sys
import numpy as np
import math
import random
import scipy
from scipy.ndimage import gaussian_filter
from easydict import EasyDict as edict

import json
import time
import copy
from PIL import Image


# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# from data_aug import data_aug_blur, data_aug_color, data_aug_noise, data_aug_light
# from data_aug import data_aug_flip, flip_data, aug_matrix
# from data_aug import show_edge, mask_to_bbox, load_json
# from data_aug import base64_2_mask, mask_2_base64, padding, Normalize_Img, Anti_Normalize_Img
# from data_aug import data_motion_blur, data_Affine, data_Perspective, data_ThinPlateSpline
# from data_aug import data_motion_blur_prior, data_Affine_prior, data_Perspective_prior, data_ThinPlateSpline_prior

class PortraitSeg(data.Dataset):
    def __init__(self, ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, exp_args):
        self.ImageRoot = ImageRoot
        self.AnnoRoot = AnnoRoot
        self.istrain = exp_args.istrain
        self.stability = exp_args.stability
        self.addEdge = exp_args.addEdge

        self.video = exp_args.video
        self.prior_prob = exp_args.prior_prob

        self.task = exp_args.task
        self.dataset = exp_args.dataset  # eg1800
        self.input_height = exp_args.input_height
        self.input_width = exp_args.input_width

        self.padding_color = exp_args.padding_color
        self.img_scale = exp_args.img_scale
        self.img_mean = exp_args.img_mean  # BGR order
        self.img_val = exp_args.img_val  # BGR order

        if self.istrain == True:
            file_object = open(ImgIds_Train, 'r')
        elif self.istrain == False:
            file_object = open(ImgIds_Test, 'r')

        try:
            self.imgIds = file_object.readlines()  # 1447
            if self.dataset == "MscocoBackground" and self.istrain == True:
                self.imgIds = self.imgIds[:5000]

            if self.dataset == "ATR" and self.istrain == True:
                self.imgIds = self.imgIds[:5000]

            # if self.istrain == False:
            #     self.imgIds = self.imgIds[:100]

        finally:
            file_object.close()
        pass

    def __getitem__(self, index):
        '''
        An item is an image. Which may contains more than one person.
        '''
        img = None
        mask = None
        bbox = None
        H = None

        if self.dataset == "supervisely":
            # basic info
            img_path = os.path.join(self.ImageRoot, self.imgIds[index].strip())
            img_name = img_path[img_path.rfind('/') + 1:]
            img = cv2.imread(img_path)

            # load mask
            annopath = img_path.replace('/img/', '/ann/')
            annopath = annopath[:annopath.find('.')] + '.json'
            ann = load_json(annopath)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            for i in range(len(ann['objects'])):
                mask_temp = np.zeros((img.shape[0], img.shape[1]))
                if ann['objects'][i]['classTitle'] == 'person_poly':
                    points = np.array(ann['objects'][i]['points']['exterior'])
                    if len(points) > 0:
                        cv2.fillPoly(mask_temp, [points], 1)
                        points = np.array(ann['objects'][i]['points']['interior'])
                        for p in points:
                            cv2.fillPoly(mask_temp, [np.array(p)], 0)
                elif ann['objects'][i]['classTitle'] == 'neutral':
                    points = np.array(ann['objects'][i]['points']['exterior'])
                    if len(points) > 0:
                        cv2.fillPoly(mask_temp, [points], 1)
                        points = np.array(ann['objects'][i]['points']['interior'])
                        for p in points:
                            cv2.fillPoly(mask_temp, [np.array(p)], 0)
                elif ann['objects'][i]['classTitle'] == 'person_bmp':
                    data = np.array(ann['objects'][i]['bitmap']['data'])
                    if data.size > 0:
                        mask_ = base64_2_mask(data)
                        origin = ann['objects'][i]['bitmap']['origin']
                        mask_temp[origin[1]:origin[1] + mask_.shape[0], origin[0]:origin[0] + mask_.shape[1]] = mask_
                mask[mask_temp > 0] = 1

            height, width, channel = img.shape
            # bbox = mask_to_bbox(mask)
            bbox = [0, 0, width - 1, height - 1]

            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                           angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height / 4)

        elif self.dataset in ["supervisely_face_easy", "supervisely_face_difficult"]:
            # basic info
            img_path = os.path.join(self.ImageRoot, self.imgIds[index].strip())
            img_name = img_path[img_path.rfind('/') + 1:]
            img = cv2.imread(img_path)
            # img = cv2.imread(img_path.replace('/img/', '/imgAug/'))

            # load mask
            annopath = img_path.replace('/img/', '/ann/')
            # annopath = img_path.replace('/img/', '/maskAug/')

            mask = cv2.imread(annopath, 0)  # origin mask = 255
            mask[mask > 0] = 1

            height, width, channel = img.shape
            # bbox = mask_to_bbox(mask)
            bbox = [0, 0, width - 1, height - 1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                           angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height / 4)

        elif self.dataset in ["flickr", "eg1800", "liveshow"]:
            # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            # img = cv2.imread(img_path.replace('Images', 'ImagesAug'))
            img_name = img_path[img_path.rfind('/') + 1:]

            # load mask
            annopath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0)
            # mask = cv2.imread(annopath.replace('Labels', 'LabelsAug'), 0)
            mask[mask > 1] = 0

            height, width, channel = img.shape
            bbox = [0, 0, width - 1, height - 1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                           angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height / 4)

        elif self.dataset in ["new_archive"]:
            # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            # img = cv2.imread(img_path.replace('Images', 'ImagesAug'))
            img_name = img_path[img_path.rfind('/') + 1:]

            # load mask

            annopath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0)
            # mask = cv2.imread(annopath.replace('Labels', 'LabelsAug'), 0)
            mask[mask > 1] = 0

            height, width, channel = img.shape
            bbox = [0, 0, width - 1, height - 1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                           angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height / 4)

        elif self.dataset in ["matting_human_sample"]:
            # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            # img = cv2.imread(img_path.replace('Images', 'ImagesAug'))
            img_name = img_path[img_path.rfind('/') + 1:]

            # load mask
            temppath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
            annopath = temppath.replace("/clip_", "/matting_")
            segmentation_image = cv2.imread(annopath, 0)
            mask = np.zeros(segmentation_image.shape, dtype=np.uint8)
            # mask = cv2.imread(annopath.replace('Labels', 'LabelsAug'), 0)
            mask[segmentation_image > 0] = 1

            height, width, channel = img.shape
            bbox = [0, 0, width - 1, height - 1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                           angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height / 4)

        elif self.dataset == "ATR":
            # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            img_name = img_path[img_path.rfind('/') + 1:]

            # load mask
            annopath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0)
            mask[mask > 1] = 1

            height, width, channel = img.shape
            bbox = [0, 0, width - 1, height - 1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                           angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height / 4)

        elif self.dataset == "MscocoBackground":
            # basic info
            img_path = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_path)
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            mask = np.zeros((height, width))

            bbox = [0, 0, width - 1, height - 1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                           angle_range=(-45, 45), scale_range=(1.5, 2.0), offset=self.input_height / 4)

        use_float_mask = False  # use original 0/1 mask as groundtruth

        # data augument: first align center to center of dst size. then rotate and scale
        if self.istrain == False:
            img_aug_ori, mask_aug_ori = padding(img, mask, size=self.input_width, padding_color=self.padding_color)

            # ===========add new channel for video stability============
            input_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            if self.video == True:
                prior = np.zeros((self.input_height, self.input_width, 1))
                input_norm = np.c_[input_norm, prior]
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = copy.deepcopy(input)
        else:
            img_aug = cv2.warpAffine(np.uint8(img), H, (self.input_width, self.input_height),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(self.padding_color, self.padding_color, self.padding_color))
            mask_aug = cv2.warpAffine(np.uint8(mask), H, (self.input_width, self.input_height),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            img_aug_ori, mask_aug_ori, aug_flag = data_aug_flip(img_aug, mask_aug)
            prior = np.zeros((self.input_height, self.input_width, 1))

            # ======== add new channel for video stability =========
            if self.video == True and self.prior_prob >= random.random():  # add new augmentation
                prior[:, :, 0] = mask_aug_ori.copy()
                prior = np.array(prior, dtype=np.float)

                if random.random() >= 0.5:
                    # modify image + mask, use groundtruth as prior
                    img_aug_ori = np.array(img_aug_ori)
                    mask_aug_ori = np.array(mask_aug_ori, dtype=np.float)
                    img_aug_ori, mask_aug_ori = data_motion_blur(img_aug_ori, mask_aug_ori)
                    img_aug_ori, mask_aug_ori = data_Affine(img_aug_ori, mask_aug_ori, self.input_height,
                                                            self.input_width, ratio=0.05)
                    img_aug_ori, mask_aug_ori = data_Perspective(img_aug_ori, mask_aug_ori, self.input_height,
                                                                 self.input_width, ratio=0.05)
                    img_aug_ori, mask_aug_ori = data_ThinPlateSpline(img_aug_ori, mask_aug_ori, self.input_height,
                                                                     self.input_width, ratio=0.05)
                    use_float_mask = True
                else:
                    # modify prior, don't change image + mask
                    prior = data_motion_blur_prior(prior)
                    prior = data_Affine_prior(prior, self.input_height, self.input_width, ratio=0.05)
                    prior = data_Perspective_prior(prior, self.input_height, self.input_width, ratio=0.05)
                    prior = data_ThinPlateSpline_prior(prior, self.input_height, self.input_width, ratio=0.05)
                    prior = prior.reshape(self.input_height, self.input_width, 1)

            # add augmentation
            img_aug = Image.fromarray(cv2.cvtColor(img_aug_ori, cv2.COLOR_BGR2RGB))
            img_aug = data_aug_color(img_aug)
            img_aug = np.asarray(img_aug)
            # img_aug = data_aug_light(img_aug)
            img_aug = data_aug_blur(img_aug)
            img_aug = data_aug_noise(img_aug)
            img_aug = np.float32(img_aug[:, :, ::-1])  # BGR, like cv2.imread

            input_norm = Normalize_Img(img_aug, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            input_ori_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)

            if self.video == True:
                input_norm = np.c_[input_norm, prior]
                input_ori_norm = np.c_[input_ori_norm, prior]

            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = np.transpose(input_ori_norm, (2, 0, 1))

        if 'seg' in self.task:
            if use_float_mask == True:
                output_mask = cv2.resize(mask_aug_ori, (self.input_width, self.input_height),
                                         interpolation=cv2.INTER_NEAREST)
                cv2.normalize(output_mask, output_mask, 0, 1, cv2.NORM_MINMAX)
                output_mask[output_mask >= 0.5] = 1
                output_mask[output_mask < 0.5] = 0
            else:
                output_mask = cv2.resize(np.uint8(mask_aug_ori), (self.input_width, self.input_height),
                                         interpolation=cv2.INTER_NEAREST)

                # add mask blur
                output_mask = np.uint8(cv2.blur(output_mask, (5, 5)))
                output_mask[output_mask >= 0.5] = 1
                output_mask[output_mask < 0.5] = 0
        else:
            output_mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8) + 255

        if self.task == 'seg':
            edge = show_edge(output_mask)
            # edge_blur = np.uint8(cv2.blur(edge, (5,5)))/255.0
            return input_ori, input, edge, output_mask

    def __len__(self):
        return len(self.imgIds)


# 3.Datasets copied from datasets.py
import torch
import torch.utils.data as data
import numpy as np


# from datasets_portraitseg import PortraitSeg
class Human(data.Dataset):
    def __init__(self, exp_args):
        assert exp_args.task in ['seg'], 'Error!, <task> should in [seg]'

        self.exp_args = exp_args
        self.task = exp_args.task
        self.datasetlist = exp_args.datasetlist  # ['EG1800']
        self.data_root = exp_args.data_root  # data_root = '/home/dongx12/Data/'
        # /home/liuchang/portraitNet/PortraitNet-master/Data/
        self.file_root = exp_args.file_root  # file_root = '/home/dongx12/PortraitNet/data/select_data/'
        # /home/liuchang/portraitNet/PortraitNet-master/data/select_data/
        self.datasets = {}
        self.imagelist = []

        # load dataset
        if 'supervisely' in self.datasetlist:
            ImageRoot = self.data_root
            AnnoRoot = self.data_root
            ImgIds_Train = self.file_root + 'supervisely_train_new.txt'
            ImgIds_Test = self.file_root + 'supervisely_test_new.txt'
            exp_args.dataset = 'supervisely'
            self.datasets['supervisely'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)

        if 'EG1800' in self.datasetlist:
            ImageRoot = self.data_root + 'EG1800/Images/'
            AnnoRoot = self.data_root + 'EG1800/Labels/'
            ImgIds_Train = self.file_root + 'eg1800_train.txt'
            ImgIds_Test = self.file_root + 'eg1800_test.txt'
            exp_args.dataset = 'eg1800'
            self.datasets['eg1800'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)

        if 'matting_human_sample' in self.datasetlist:
            ImageRoot = self.data_root + 'matting_human_sample/clip_img/'
            AnnoRoot = self.data_root + 'matting_human_sample/matting/'
            ImgIds_Train = self.file_root + 'matting_human_train.txt'
            ImgIds_Test = self.file_root + 'matting_human_test.txt'
            exp_args.dataset = 'matting_human_sample'
            self.datasets['matting_human_sample'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test,
                                                                self.exp_args)
        if 'new_archive' in self.datasetlist:
            ImageRoot = self.data_root + 'new_archive/images/'
            AnnoRoot = self.data_root + 'new_archive/annotations/'
            ImgIds_Train = self.file_root + 'new_archive_train.txt'
            ImgIds_Test = self.file_root + 'new_archive_test.txt'
            exp_args.dataset = 'new_archive'
            self.datasets['new_archive'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)

        if 'ATR' in self.datasetlist:
            ImageRoot = self.data_root + 'ATR/train/images/'
            AnnoRoot = self.data_root + 'ATR/train/seg/'
            ImgIds_Train = self.file_root + 'ATR_train.txt'
            ImgIds_Test = self.file_root + 'ATR_test.txt'
            exp_args.dataset = 'ATR'
            self.datasets['ATR'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)

        if 'supervisely_face_easy' in self.datasetlist:
            ImageRoot = self.data_root
            AnnoRoot = self.data_root
            ImgIds_Train = self.file_root + 'supervisely_face_train_easy.txt'
            ImgIds_Test = self.file_root + 'supervisely_face_test_easy.txt'
            exp_args.dataset = 'supervisely_face_easy'
            self.datasets['supervisely_face_easy'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test,
                                                                 self.exp_args)

        if 'supervisely_face_difficult' in self.datasetlist:
            ImageRoot = self.data_root
            AnnoRoot = self.data_root
            ImgIds_Train = self.file_root + 'supervisely_face_train_difficult.txt'
            ImgIds_Test = self.file_root + 'supervisely_face_test_difficult.txt'
            exp_args.dataset = 'supervisely_face_difficult'
            self.datasets['supervisely_face_difficult'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test,
                                                                      self.exp_args)

        if 'MscocoBackground' in self.datasetlist:
            dataType = 'train2017'
            ImageRoot = self.data_root
            AnnoRoot = self.data_root + 'mscoco2017/annotations/person_keypoints_{}.json'.format(dataType)
            ImgIds_Train = self.file_root + 'select_mscoco_background_train2017.txt'
            ImgIds_Test = self.file_root + 'select_mscoco_background_val2017.txt'
            exp_args.dataset = 'MscocoBackground'
            self.datasets['background'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)

        # image list
        for key in self.datasets.keys():
            length = len(self.datasets[key])
            for i in range(length):  # eg1800 1447
                self.imagelist.append([key, i])

    def __getitem__(self, index):
        subset, subsetidx = self.imagelist[index]

        if self.task == 'seg':
            input_ori, input, output_edge, output_mask = self.datasets[subset][subsetidx]
            return input_ori.astype(np.float32), input.astype(np.float32), \
                output_edge.astype(np.int64), output_mask.astype(np.int64)

    def __len__(self):
        return len(self.imagelist)
