# portraitnet.py

# This file defines the PortraitNet model architecture.
# This file should include:

# 1. Import necessary libraries
#    - PyTorch and its neural network modules
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable

import numpy as np


def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # print filt
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w


# 1x1 Convolution
def conv_1x1(inp, oup):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )


def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=kernel, stride=stride, padding=int((kernel - 1) / 2),
                  bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel, stride, int((kernel - 1) / 2), groups=inp, bias=False),
        nn.BatchNorm2d(num_features=inp, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            conv_dw(inp, oup, 3, stride=stride),
            nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=3, stride=1, padding=1, groups=oup, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=oup, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )
        if inp == oup:
            self.residual = None
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.block(x)
        if self.residual is not None:
            residual = self.residual(x)

        out += residual
        out = self.relu(out)
        return out


class MobileNetV2(nn.Module):
    def __init__(self, n_class=2, useUpsample=False, useDeconvGroup=False, addEdge=False,
                 channelRatio=1.0, minChannel=16, weightInit=True, video=False):
        super(MobileNetV2, self).__init__()
        '''
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        '''
        self.addEdge = addEdge
        self.channelRatio = channelRatio
        self.minChannel = minChannel
        self.useDeconvGroup = useDeconvGroup

        if video == True:
            self.stage0 = conv_bn(4, self.depth(32), 3, 2)
        else:
            self.stage0 = conv_bn(3, self.depth(32), 3, 2)

        self.stage1 = InvertedResidual(self.depth(32), self.depth(16), 1, 1)  # 1/2

        self.stage2 = nn.Sequential(  # 1/4
            InvertedResidual(self.depth(16), self.depth(24), 2, 6),
            InvertedResidual(self.depth(24), self.depth(24), 1, 6),
        )

        self.stage3 = nn.Sequential(  # 1/8
            InvertedResidual(self.depth(24), self.depth(32), 2, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
        )

        self.stage4 = nn.Sequential(  # 1/16
            InvertedResidual(self.depth(32), self.depth(64), 2, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
        )

        self.stage5 = nn.Sequential(  # 1/16
            InvertedResidual(self.depth(64), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
        )

        self.stage6 = nn.Sequential(  # 1/32
            InvertedResidual(self.depth(96), self.depth(160), 2, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
        )

        self.stage7 = InvertedResidual(self.depth(160), self.depth(320), 1, 6)  # 1/32

        if useUpsample == True:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv2 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv3 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv4 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv5 = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            if useDeconvGroup == True:
                self.deconv1 = nn.ConvTranspose2d(self.depth(96), self.depth(96), groups=self.depth(96),
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv2 = nn.ConvTranspose2d(self.depth(32), self.depth(32), groups=self.depth(32),
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv3 = nn.ConvTranspose2d(self.depth(24), self.depth(24), groups=self.depth(24),
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv4 = nn.ConvTranspose2d(self.depth(16), self.depth(16), groups=self.depth(16),
                                                  kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv5 = nn.ConvTranspose2d(self.depth(8), self.depth(8), groups=self.depth(8),
                                                  kernel_size=4, stride=2, padding=1, bias=False)
            else:
                self.deconv1 = nn.ConvTranspose2d(self.depth(96), self.depth(96),
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv2 = nn.ConvTranspose2d(self.depth(32), self.depth(32),
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv3 = nn.ConvTranspose2d(self.depth(24), self.depth(24),
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv4 = nn.ConvTranspose2d(self.depth(16), self.depth(16),
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)
                self.deconv5 = nn.ConvTranspose2d(self.depth(8), self.depth(8),
                                                  groups=1, kernel_size=4, stride=2, padding=1, bias=False)

        self.transit1 = ResidualBlock(self.depth(320), self.depth(96))
        self.transit2 = ResidualBlock(self.depth(96), self.depth(32))
        self.transit3 = ResidualBlock(self.depth(32), self.depth(24))
        self.transit4 = ResidualBlock(self.depth(24), self.depth(16))
        self.transit5 = ResidualBlock(self.depth(16), self.depth(8))

        self.pred = nn.Conv2d(self.depth(8), n_class, 3, 1, 1, bias=False)
        if self.addEdge == True:
            self.edge = nn.Conv2d(self.depth(8), n_class, 3, 1, 1, bias=False)

        if weightInit == True:
            self._initialize_weights()

    def depth(self, channels):
        min_channel = min(channels, self.minChannel)
        return max(min_channel, int(channels * self.channelRatio))

    def forward(self, x):
        feature_1_2 = self.stage0(x)
        feature_1_2 = self.stage1(feature_1_2)
        feature_1_4 = self.stage2(feature_1_2)
        feature_1_8 = self.stage3(feature_1_4)
        feature_1_16 = self.stage4(feature_1_8)
        feature_1_16 = self.stage5(feature_1_16)
        feature_1_32 = self.stage6(feature_1_16)
        feature_1_32 = self.stage7(feature_1_32)

        up_1_16 = self.deconv1(self.transit1(feature_1_32))
        up_1_8 = self.deconv2(self.transit2(feature_1_16 + up_1_16))
        up_1_4 = self.deconv3(self.transit3(feature_1_8 + up_1_8))
        up_1_2 = self.deconv4(self.transit4(feature_1_4 + up_1_4))
        up_1_1 = self.deconv5(self.transit5(up_1_2))

        pred = self.pred(up_1_1)
        if self.addEdge == True:
            edge = self.edge(up_1_1)
            return pred, edge
        else:
            return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = make_bilinear_weights(m.kernel_size[0], m.out_channels)  # same as caffe
                m.weight.data.copy_(initial_weight)
                if self.useDeconvGroup == True:
                    m.requires_grad = False  # use deconvolution as bilinear upsample
                    print("freeze deconv")
        pass

# 2. Define the PortraitNet class
import numpy as np
import torch
import torch.nn as nn

# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn8s.py
# Edit: Added a dropout parameter to the class definition
class FCN8s(nn.Module):

    def __init__(self, n_class=21, dropout=0.5):
        super(FCN8s, self).__init__()
        self.name = "fcn8s"
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=dropout)

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=dropout)

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


class PortraitFCN(FCN8s):
    def __init__(self, n_class=2, dropout=0.5):
        super(PortraitFCN, self).__init__(n_class=n_class, dropout=dropout)
        self.name = "pfcn"
        self.n_class = n_class


class PortraitFCNPlus(PortraitFCN):
    def __init__(self, load_weights=False, dropout=0.5):
        super(PortraitFCNPlus, self).__init__(dropout=dropout)
        self.name = "pfcnp"
        if load_weights:
            path_to_weights = "portraitseg/weights/portraitfcn_untrained.pth"
            self.load_state_dict(torch.load(path_to_weights))
        kernels = self.conv1_1.weight.data
        k = 3  # Kernel size
        c_in = 6  # Number of input channels
        c_out = 64  # Number of output channels
        size = (c_in, k, k)
        n = k * k * c_out
        superkernels = torch.from_numpy(np.zeros((c_out, c_in, k, k)))
        for i, kernel in enumerate(kernels):
            superkernel = torch.from_numpy(np.random.normal(0, np.sqrt(2./n),
                                                            size=size))
            superkernel[:3] = kernel
            superkernels[i] = superkernel
        self.conv1_1.weight.data = superkernels.float()


class FCN8s_probe(FCN8s):
    """Outputs activations from multiple layers."""
    def __init__(self, n_class=21):
        super(FCN8s_probe, self).__init__()

    def forward(self, x):
        activations = []

        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8
        activations.append(('pool3', pool3))

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16
        activations.append(('pool4', pool4))

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        activations.append(('fr', h))

        h = self.score_fr(h)  # 1/32
        activations.append(('score_fr', h))
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        activations.append(('upscore2', upscore2))

        h = self.score_pool4(pool4)
        activations.append(('score_pool4', h))
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        activations.append(('score_pool3', h))
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        activations.append(('output', h))

        return activations