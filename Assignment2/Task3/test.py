import os
import os.path as osp

import numpy as np
import torch
from block_matching import add_padding, visualize_disparity
from dataset import KITTIDataset
from siamese_neural_network import StereoMatchingNetwork, calculate_similarity_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def compute_disparity_CNN(infer_similarity_metric, img_l, img_r, max_disparity=50):
    """
    Computes the disparity of the stereo image pair.

    Args:
        infer_similarity_metric:  pytorch module object
        img_l: tensor holding the left image (height,width,channels)
        img_r: tensor holding the right image (height,width,channels)
        max_disparity (int): maximum disparity

    Returns:
        D: tensor holding the disparity
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 8)
    # -------------------------------------
    img_l = img_l.to(infer_similarity_metric.device)
    img_r = img_r.to(infer_similarity_metric.device)
    height, width, channels = img_l.shape
    D = torch.zeros((height, width), device=img_l.device)  # 初始化视差图

    # 为了方便计算，将图像扩展一个批次维度，变成 (1, height, width, channels)
    img_l = img_l.unsqueeze(0)  # (1, height, width, channels)
    img_r = img_r.unsqueeze(0)  # (1, height, width, channels)
    score_map = torch.full((height - 8, width - 8), -np.inf, device=img_l.device)
    # 逐视差值计算相似度分数
    for d in range(1, max_disparity + 1):
        # 向右偏移右图像块以生成视差块
        img_r_shifted = torch.zeros_like(img_r)
        img_r_shifted[:, :, d:, :] = img_r[:, :, :-d, :]

        # 计算当前视差的相似度分数
        similarity_score = calculate_similarity_score(infer_similarity_metric, img_l, img_r_shifted)
        similarity_score = similarity_score.squeeze(0)  # (height-8, width-8)

        # Find the area to change
        update_mask = similarity_score > score_map # (height-8, width-8)
        score_map[update_mask] = similarity_score[update_mask]

        # 更新视差图：如果当前视差的相似度更大，则更新视差
        D[4:height-4, 4:width-4][update_mask] = d
        rate = d / (max_disparity+1)
        print('\rProgress: {:.2f}%'.format(rate * 100), end='')
    return D

def main():
    # Hyperparameters
    training_iterations = 5000
    batch_size = 32
    learning_rate = 3e-4
    patch_size = 9
    padding = patch_size // 2
    max_disparity = 50

    # Shortcuts for directories
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "dataset/KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network", f"iteration_{training_iterations}")
    model_path = osp.join(out_dir, "best_model.pth")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Set network to eval mode
    infer_similarity_metric = StereoMatchingNetwork()
    infer_similarity_metric.load_state_dict(torch.load(model_path))
    infer_similarity_metric.eval()
    infer_similarity_metric.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # infer_similarity_metric.to('cpu')

    # Load KITTI test split
    dataset = KITTIDataset(osp.join(data_dir, "testing"))
    # Loop over test images
    for i in range(len(dataset)):
        print(f"\nProcessing {i} image")
        # Load images and add padding
        img_left, img_right = dataset[i]
        img_left_padded, img_right_padded = add_padding(img_left, padding), add_padding(
            img_right, padding
        )
        img_left_padded, img_right_padded = torch.Tensor(img_left_padded), torch.Tensor(
            img_right_padded
        )

        disparity_map = compute_disparity_CNN(
            infer_similarity_metric,
            img_left_padded,
            img_right_padded,
            max_disparity=max_disparity,
        ).to('cpu')
        # Visulization
        title = (
            f"Disparity map for image {i} with SNN (training iterations {training_iterations}, "
            f"batch size {batch_size}, patch_size {patch_size})"
        )
        file_name = f"{i}_training_iterations_{training_iterations}.png"
        out_file_path = osp.join(out_dir, file_name)
        visualize_disparity(
            disparity_map.squeeze(),
            img_left.squeeze(),
            img_right.squeeze(),
            out_file_path,
            title,
            max_disparity=max_disparity,
        )


if __name__ == "__main__":
    main()
