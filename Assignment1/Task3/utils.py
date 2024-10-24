# utils.py
import numpy as np
# This file contains utility functions for the project.
# This file should include:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# - Loss computation (e.g., custom loss functions)
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# - Metrics computation (e.g., IoU, mIoU)
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

# - Logging and visualization tools
import tensorflow as tf
import numpy as np

# import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

from PIL import Image


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
            #             scipy.misc.toimage(img).save(s, format="png")
            Image.fromarray((img * 255).astype(np.uint8)).save(s, format='png')
            # Create an Image object
            #     img_sum = tf.summary.Image(encoded_image_string=s.getvalue(),
            #                                height=img.shape[0],
            #                                width=img.shape[1])
            #     # Create a Summary value
            #     img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
            #
            # # Create and write Summary
            # summary = tf.summary(value=img_summaries)
            # self.writer.add_summary(summary, step)
            img_str = s.getvalue()

            # TensorFlow 2.x 中不再使用 tf.Summary.Image，改用 tf.summary.image
            # 替换原来的 tf.Summary.Image 逻辑
            tf_image = tf.image.decode_png(img_str, channels=3)  # 将字节流解码为图片
            tf.summary.image(f'{tag}/{i}', tf.expand_dims(tf_image, 0), step=step)

        # TensorFlow 2.x 不再使用 tf.Summary，因此不需要创建和写入 Summary
        # summary = tf.Summary(value=img_summaries)
        # self.writer.add_summary(summary, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, hist, step=step)
            self.writer.flush()
