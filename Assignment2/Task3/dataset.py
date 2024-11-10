#!/usr/bin/env python

import time
from os import listdir, path
from threading import Lock, Thread

import numpy as np
import torch
from imageio import imread
from skimage.transform import rescale


# convert rgb image to grayscale
def rgb2gray(I):
    return np.dot(I[..., :3], [0.299, 0.587, 0.114])


class KITTIDataset(object):
    """Prepare KITTI dataset"""

    def __init__(self, image_dir, disparity_dir=None, downsample=True):
        self.disparity_dir = disparity_dir
        self.downsample = downsample

        left_dir = path.join(image_dir, "image_2")
        right_dir = path.join(image_dir, "image_3")

        self._left_images = sorted(
            [path.join(left_dir, img) for img in listdir(left_dir) if "_10." in img]
        )
        self._right_images = sorted(
            [path.join(right_dir, img) for img in listdir(right_dir) if "_10." in img]
        )
        assert len(self._left_images) == len(self._right_images)

        if disparity_dir is not None:
            self._disp_images = sorted(
                [path.join(disparity_dir, img) for img in listdir(disparity_dir)]
            )
            assert len(self._left_images) == len(self._disp_images)
        else:
            self._disp_images = []

        print("KITTI data loaded (%d images)!" % len(self._left_images))

    def __len__(self):
        return len(self._left_images)

    def __getitem__(self, i):
        img_l = imread(self._left_images[i]).astype(np.float32) / 255.0
        img_r = imread(self._right_images[i]).astype(np.float32) / 255.0

        img_l = rgb2gray(img_l)[..., np.newaxis]
        img_r = rgb2gray(img_r)[..., np.newaxis]

        if self.downsample:
            img_l = rescale(
                img_l, 0.5, mode="reflect", anti_aliasing=True, multichannel=True
            )
            img_r = rescale(
                img_r, 0.5, mode="reflect", anti_aliasing=True, multichannel=True
            )

        if self.disparity_dir is not None:
            disp = imread(self._disp_images[i]).astype(np.float32) / 256.0

            if self.downsample:
                H, W = disp.shape
                disp = disp[np.arange(0, H, 2), :]  # Downsample first dimension
                disp = disp[:, np.arange(0, W, 2)]  # Downsample second dimension
                disp = disp / 2.0  # Scale values accordingly

            disp[disp <= 0] = -1

            return img_l, img_r, disp
        else:
            return img_l, img_r


class PatchProvider(object):
    """Provide training patches"""

    def __init__(self, data, patch_size=(7, 7), N=(4, 10), P=1):
        self._data = data
        self._patch_size = patch_size
        self._N = N
        self._P = P
        self.idxs = None

        self._stop = False
        self._cache = 5
        self._lock = Lock()

    def _get_neg_idx(self, col, W):
        # local copy for convenience
        half_patch = self._patch_size[1] // 2
        N = self._N

        neg_offset = np.random.randint(N[0], N[1] + 1)
        neg_offset = neg_offset * np.sign(np.random.rand() - 0.5).astype(np.int32)

        if half_patch <= col + neg_offset < W - half_patch:
            return slice(
                col + neg_offset - half_patch, col + neg_offset + half_patch + 1
            )
        else:
            return self._get_neg_idx(col, W)

    def _get_pos_idx(self, col, W):
        # local copy for convenience
        half_patch = self._patch_size[1] // 2
        P = self._P

        pos_offset = np.random.randint(-P, P + 1)
        if half_patch <= col + pos_offset < W - half_patch:
            return slice(
                col + pos_offset - half_patch, col + pos_offset + half_patch + 1
            )
        else:
            return self._get_pos_idx(col, W)

    def random_patch(self):
        # local copy for convenience
        patch_size = self._patch_size
        half_patch = np.array(patch_size) // 2
        img_l, img_r, disp = self._data[int(np.random.rand() * len(self._data))]
        H, W = img_l.shape[:2]
        while True:
            half_p = patch_size[0] // 2
            row = np.random.randint(half_p, H - half_p)
            col = np.random.randint(half_p, W - half_p)
            d = disp[row, col]
            if d > 0 and (col - d) > half_p and (col - d) < W - half_p:
                break

        ref_idx = (
            slice(row - half_patch[0], row + half_patch[0] + 1),
            slice(col - half_patch[1], col + half_patch[1] + 1),
        )
        neg_idx = (
            slice(row - half_patch[0], row + half_patch[0] + 1),
            self._get_neg_idx(int(col - disp[row, col]), W),
        )
        pos_idx = (
            slice(row - half_patch[0], row + half_patch[0] + 1),
            self._get_pos_idx(int(col - disp[row, col]), W),
        )
        return img_l[ref_idx], img_r[pos_idx], img_r[neg_idx]

    def iterate_batches(self, batch_size):
        # Get a patch to infer the image shape

        patch = self.random_patch()
        channels = patch[0].shape[-1]

        ref_batch = np.zeros(
            (self._cache * batch_size,) + self._patch_size + (channels,),
            dtype="float32",
        )
        pos_batch = np.zeros_like(ref_batch)
        neg_batch = np.zeros_like(ref_batch)

        # start the thread
        self._thread = Thread(
            target=self.fill_batches, args=(ref_batch, pos_batch, neg_batch)
        )
        self._stop = False
        self._thread.start()

        # wait for the buffers to fill
        while True:
            time.sleep(1)
            with self._lock:
                if ref_batch[-1].sum() == 0:
                    pass
                else:
                    break

        # start generating batches
        while True:
            self.idxs = np.random.choice(len(ref_batch), batch_size)
            with self._lock:
                yield torch.Tensor(ref_batch[self.idxs]), torch.Tensor(
                    pos_batch[self.idxs]
                ), torch.Tensor(neg_batch[self.idxs])

    def fill_batches(self, ref, pos, neg):
        idx = 0
        while not self._stop:
            patch = self.random_patch()
            with self._lock:
                ref[idx] = patch[0]
                pos[idx] = patch[1]
                neg[idx] = patch[2]
            idx += 1
            idx = idx % len(ref)

    def stop(self):
        self._stop = True
        self._thread.join()
