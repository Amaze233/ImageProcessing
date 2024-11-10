import os
import os.path as osp

import cv2
import numpy as np
from dataset import KITTIDataset
from matplotlib import pyplot as plt


def add_padding(I, padding):
    """
    Adds zero padding to an RGB or grayscale image.

    Args:
        I (np.ndarray): HxWx? numpy array containing RGB or grayscale image

    Returns:
        P (np.ndarray): (H+2*padding)x(W+2*padding)x? numpy array containing zero padded image
    """
    if len(I.shape) == 2:
        H, W = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding), dtype=np.float32)
        padded[padding:-padding, padding:-padding] = I
    else:
        H, W, C = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding, C), dtype=I.dtype)
        padded[padding:-padding, padding:-padding] = I

    return padded


def sad(image_left, image_right, window_size=3, max_disparity=50):
    """
    Compute the sum of absolute differences between image_left and image_right.

    Args:
        image_left (np.ndarray): HxW numpy array containing grayscale right image
        image_right (np.ndarray): HxW numpy array containing grayscale left image
        window_size: window size (default 3)
        max_disparity: maximal disparity to reduce search range (default 50)

    Returns:
        D (np.ndarray): HxW numpy array containing the disparity for each pixel
    """

    D = np.zeros_like(image_left)

    # add zero padding
    padding = window_size // 2
    image_left = add_padding(image_left, padding).astype(np.float32)
    image_right = add_padding(image_right, padding).astype(np.float32)

    height = image_left.shape[0]
    width = image_left.shape[1]

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 1)
    # -------------------------------------
    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            best_sad = float('inf')
            best_disparity = 0

            for d in range(max_disparity + 1):
                # Ensure we don't go out of bounds
                if x - d < padding or x - d >= width - padding:
                    break

                # Extract the matching blocks
                block_left = image_left[y - padding:y + padding + 1, x - padding:x + padding + 1]
                block_right = image_right[y - padding:y + padding + 1, x - padding - d:x + padding + 1 - d]

                # Compute SAD
                sad_value = np.sum(np.abs(block_left - block_right))

                if sad_value < best_sad:
                    best_sad = sad_value
                    best_disparity = d

            D[y - padding, x - padding] = best_disparity
        rate = y / height
        print('\rProgress: {:.2f}%'.format(rate * 100), end='')
    return D


def visualize_disparity(
        disparity, im_left, im_right, out_file_path, title="Disparity Map", max_disparity=50
):
    """
    Generates a visualization for the disparity map.

    Args:
        disparity (np.array): disparity map
        im_left (np.array): left image
        im_right (np.array): right image
        out_file_path: output file path
        title: plot title
        max_disparity: maximum disparity
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 2)
    # -------------------------------------
    plt.figure(figsize=(12, 10))

    # Display left image
    plt.subplot(3, 1, 1)
    plt.title("Left Image")
    plt.imshow(im_left, cmap='gray')
    plt.axis('off')

    # Display right image
    plt.subplot(3, 1, 2)
    plt.title("Right Image")
    plt.imshow(im_right, cmap='gray')
    plt.axis('off')

    # Display disparity map
    plt.subplot(3, 1, 3)
    plt.title(title)
    plt.imshow(disparity, cmap='jet_r', vmin=0, vmax=max_disparity)
    # plt.colorbar(label='Disparity')
    plt.axis('off')

    # Save the visualization
    plt.tight_layout()
    plt.savefig(out_file_path, bbox_inches='tight', pad_inches=0.1)

    # Individual save the disparity plot
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.imshow(disparity, cmap='jet_r', vmin=0, vmax=max_disparity)
    plt.colorbar(label='Disparity', shrink=0.60, pad=0.02)
    plt.axis('off')
    plt.savefig(out_file_path.replace('.png', '_colorbar.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    # Hyperparameters
    window_size = 3
    max_disparity = 50

    # Shortcuts
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "dataset/KITTI_2015_subset")
    out_dir = osp.join(
        root_dir, "output/handcrafted_stereo", f"window_size_{window_size}"
    )
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)

    # Load dataset
    dataset = KITTIDataset(osp.join(data_dir, "testing"))

    # Calculation and Visualization
    for i in range(len(dataset)):
        # Load left and right images
        im_left, im_right = dataset[i]
        im_left, im_right = im_left.squeeze(-1), im_right.squeeze(-1)
        print(f"Processing image {i}...")
        # Calculate disparity
        D = sad(im_left, im_right, window_size=window_size, max_disparity=max_disparity)

        # Define title and output file name for the plot
        title = f"Disparity map for image {i} with block matching (window size {window_size})"
        out_file_path = osp.join(out_dir, f"{i}_w{window_size}.png")

        # Visualize the disparty and save it to a file
        visualize_disparity(
            D,
            im_left,
            im_right,
            out_file_path,
            title=title,
            max_disparity=max_disparity,
        )
        print("\nDone!")


if __name__ == "__main__":
    main()
