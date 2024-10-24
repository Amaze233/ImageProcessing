# test.py
import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable

import os
import cv2
import numpy as np
from easydict import EasyDict as edict
from yaml import load

import matplotlib
import matplotlib.pyplot as plt
import sys

from Task3.dataset import Normalize_Img
from Task3.portraitnet import MobileNetV2


# This file is used for evaluating the trained model on test data.
# This file should include:

# 1. Model loading
def padding_img(img_ori, size=224, color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]
    img = np.zeros((max(height, width), max(height, width), 3)) + color

    if (height > width):
        padding = int((height - width) / 2)
        img[:, padding:padding + width, :] = img_ori
    else:
        padding = int((width - height) / 2)
        img[padding:padding + height, :, :] = img_ori

    img = np.uint8(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.array(img, dtype=np.float32)


def resize_padding(image, dstshape, padValue=0):
    height, width, _ = image.shape
    ratio = float(width) / height  # ratio = (width:height)
    dst_width = int(min(dstshape[1] * ratio, dstshape[0]))
    dst_height = int(min(dstshape[0] / ratio, dstshape[1]))
    origin = [int((dstshape[1] - dst_height) / 2), int((dstshape[0] - dst_width) / 2)]
    if len(image.shape) == 3:
        image_resize = cv2.resize(image, (dst_width, dst_height))
        newimage = np.zeros(shape=(dstshape[1], dstshape[0], image.shape[2]), dtype=np.uint8) + padValue
        newimage[origin[0]:origin[0] + dst_height, origin[1]:origin[1] + dst_width, :] = image_resize
        bbx = [origin[1], origin[0], origin[1] + dst_width, origin[0] + dst_height]  # x1,y1,x2,y2
    else:
        image_resize = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_NEAREST)
        newimage = np.zeros(shape=(dstshape[1], dstshape[0]), dtype=np.uint8)
        newimage[origin[0]:origin[0] + height, origin[1]:origin[1] + width] = image_resize
        bbx = [origin[1], origin[0], origin[1] + dst_width, origin[0] + dst_height]  # x1,y1,x2,y2
    return newimage, bbx


def generate_input(exp_args, inputs, prior=None):
    inputs_norm = Normalize_Img(inputs, scale=exp_args.img_scale, mean=exp_args.img_mean, val=exp_args.img_val)

    if exp_args.video == True:
        if prior is None:
            prior = np.zeros((exp_args.input_height, exp_args.input_width, 1))
            inputs_norm = np.c_[inputs_norm, prior]
        else:
            prior = prior.reshape(exp_args.input_height, exp_args.input_width, 1)
            inputs_norm = np.c_[inputs_norm, prior]

    inputs = np.transpose(inputs_norm, (2, 0, 1))
    return np.array(inputs, dtype=np.float32)


def pred_single(model, exp_args, img_ori, prior=None):
    model.eval()
    softmax = nn.Softmax(dim=1)

    in_shape = img_ori.shape
    img, bbx = resize_padding(img_ori, [exp_args.input_height, exp_args.input_width], padValue=exp_args.padding_color)

    in_ = generate_input(exp_args, img, prior)
    in_ = in_[np.newaxis, :, :, :]

    if exp_args.addEdge == True:
        output_mask, output_edge = model(Variable(torch.from_numpy(in_)).cuda())
    else:
        output_mask = model(Variable(torch.from_numpy(in_)).cuda())
    prob = softmax(output_mask)
    pred = prob.data.cpu().numpy()

    predimg = pred[0].transpose((1, 2, 0))[:, :, 1]
    out = predimg[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    out = cv2.resize(out, (in_shape[1], in_shape[0]))
    return out, predimg

def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1 > 0] = 1
    sum2 = img + mask
    sum2[sum2 < 2] = 0
    sum2[sum2 >= 2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0 * np.sum(sum2) / np.sum(sum1)


# ============Load the trained model from a checkpoint==================
print('===========> loading config <============')
config_path = 'config/model_matting_human_test.yaml'
print("config path: ", config_path)
with open(config_path, 'rb') as f:
    cont = f.read()
cf = yaml.load(cont, Loader=yaml.FullLoader)
print('finish load config file ...')
exp_args = edict()
exp_args.istrain = False
exp_args.task = cf['task']
exp_args.datasetlist = cf['datasetlist']  # ['EG1800', ATR', 'MscocoBackground', 'supervisely_face_easy']

exp_args.model_root = cf['model_root']
exp_args.data_root = cf['data_root']
exp_args.file_root = cf['file_root']

# the height of input images, default=224
exp_args.input_height = 224
# the width of input images, default=224
exp_args.input_width = 224

# if exp_args.video=True, add prior channel for input images, default=False
exp_args.video = False
# the probability to set empty prior channel, default=0.5
exp_args.prior_prob = 0.5

# whether to add boundary auxiliary loss, default=False
exp_args.addEdge = False
# the weight of boundary auxiliary loss, default=0.1
exp_args.edgeRatio = 0.1
# whether to add consistency constraint loss, default=False
exp_args.stability = False
# whether to use KL loss in consistency constraint loss, default=True
exp_args.use_kl = True
# temperature in consistency constraint loss, default=1
exp_args.temperature = 1
# the weight of consistency constraint loss, default=2
exp_args.alpha = 2

# input normalization parameters
exp_args.padding_color = 128
exp_args.img_scale = 1
# BGR order, image mean, default=[103.94, 116.78, 123.68]
exp_args.img_mean = [103.94, 116.78, 123.68]
# BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
exp_args.img_val = [0.017, 0.017, 0.017]

# whether to use pretian model to init portraitnet,default true
exp_args.init = True
# whether to continue training
exp_args.resume = False

# if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
exp_args.useUpsample = False
# if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
exp_args.useDeconvGroup = False

# set training dataset
exp_args.istrain = True

print ('===========> loading model <===========')
netmodel_video = MobileNetV2(n_class=2,
                             useUpsample=exp_args.useUpsample,
                             useDeconvGroup=exp_args.useDeconvGroup,
                             addEdge=exp_args.addEdge,
                             channelRatio=1.0,
                             minChannel=16,
                             weightInit=True,
                             video=exp_args.video).cuda()

bestModelFile = os.path.join(exp_args.model_root, 'model_best-100.pth.tar')
if os.path.isfile(bestModelFile):
    checkpoint = torch.load(bestModelFile, weights_only=False)
    # print(checkpoint)
    netmodel_video.load_state_dict(checkpoint['state_dict'])
    print("minLoss: ", checkpoint['minLoss'], checkpoint['epoch'])
    print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(bestModelFile))

# 2.Test model by image and video


# ==========Image Test===========
print("==========Image Test===========")
img_ori = cv2.imread("data/images/img_4.jpg")
# mask_ori = cv2.imread("/home/dongx12/Data/EG1800/Labels/00457.png")

prior = None
height, width, _ = img_ori.shape
# 背景模糊
background = img_ori.copy()
background = cv2.blur(background, (30, 30))

alphargb, pred = pred_single(netmodel_video, exp_args, img_ori, prior)
alphargb = cv2.cvtColor(alphargb, cv2.COLOR_GRAY2BGR)
result = np.uint8(img_ori * alphargb)

myImg = np.ones((height, width * 4 + 60, 3)) * 255
myImg[:, :width, :] = img_ori
myImg[:, width + 20:width * 2 + 20, :] = alphargb*255
myImg[:, width * 2 + 40:width * 3 + 40, :] = result
myImg[:, width * 3 + 60:width * 4 + 60, :] = np.uint8(result + background * (1 - alphargb))

plt.imshow(myImg[:, :, ::-1] / 255)
plt.yticks([])
plt.xticks([])
plt.savefig('results/img_4_mh-100.png', bbox_inches='tight', pad_inches=0)
plt.show()
# ==========Video Test===========
print("==========Video Test===========")
videofile = 'data/douyu_origin.mp4'
videoCapture = cv2.VideoCapture(videofile)
size = ((int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) + 20) * 3, int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('data/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

success, frame = videoCapture.read()
cnt = 1
while success:
    if cnt == 1:
        prior = None  # first frame
    else:
        prior = pred_video

    alpha_video, pred_video = pred_single(netmodel_video, exp_args, frame, prior)
    alpha_image, pred_image = pred_single(netmodel_video, exp_args, frame, None)


    def blend(frame, alpha):
        background = np.zeros(frame.shape) + [255, 255, 255]
        alphargb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        result = np.uint8(frame * alphargb + background * (1 - alphargb))
        return frame, alphargb * 255, result


    _, alphargb_video, _ = blend(frame, alpha_video)
    _, alphargb_image, _ = blend(frame, alpha_image)

    padding = np.ones((frame.shape[0], 20, 3), dtype=np.uint8) * 255
    result = np.uint8(np.hstack((frame, padding,
                                 alphargb_video, padding,
                                 alphargb_image, padding)))
    videoWriter.write(result)
    success, frame = videoCapture.read()
    cnt += 1

    if cnt % 100 == 0:
        print("cnt: ", cnt)

videoWriter.release()

print("finish")


