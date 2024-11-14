import os
import os.path as osp
import time

import numpy as np
import torch
from tqdm import tqdm

from dataset import KITTIDataset, PatchProvider
from siamese_neural_network import StereoMatchingNetwork, calculate_similarity_score


def hinge_loss(score_pos, score_neg, label):
    """
    Computes the hinge loss for the similarity of a positive and a negative example.

    Args:
        score_pos (torch.Tensor): similarity score of the positive example
        score_neg (torch.Tensor): similarity score of the negative example
        label (torch.Tensor): the true labels

    Returns:
        avg_loss (torch.Tensor): the mean loss over the patch and the mini batch
        acc (torch.Tensor): the accuracy of the prediction
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 6)
    # -------------------------------------
    # Hinge Loss: max(0, 1 - score_pos + score_neg)
    # Accuracy: 1 if score_pos > score_neg else 0
    margin = 1  # or 0.2
    loss = torch.clamp(margin + score_neg - score_pos, min=0)
    avg_loss = torch.mean(loss)
    correct_prediction = (score_pos > score_neg).float()
    acc = torch.mean((correct_prediction == label).float())

    return avg_loss, acc


def training_loop(
        infer_similarity_metric,
        patches,
        optimizer,
        out_dir,
        start_iteration=1,
        iterations=1000,
        batch_size=128,
):
    """
    Runs the training loop of the siamese network.

    Args:
        infer_similarity_metric (obj): pytorch module
        patches (obj): patch provider object
        optimizer (obj): optimizer object
        out_dir (str): output file directory
        iterations (int): number of iterations to perform
        batch_size (int): batch size
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 6)
    # -------------------------------------
    best_loss = float('inf')
    best_model_path = os.path.join(out_dir, "best_model.pth")
    checkpoint_path = os.path.join(out_dir, "checkpoint.pth")
    batch_generator = patches.iterate_batches(batch_size)

    for i in range(start_iteration, iterations + 1):
        # Get a batch of patches
        ref_batch, pos_batch, neg_batch = next(batch_generator)

        # Pass Forward and calculate score
        pos_score = calculate_similarity_score(infer_similarity_metric, ref_batch, pos_batch).cuda()
        neg_score = calculate_similarity_score(infer_similarity_metric, ref_batch, neg_batch).cuda()

        # Compute loss and accuracy
        labels = torch.ones_like(pos_score)
        loss, acc = hinge_loss(pos_score, neg_score, labels)

        # Pass Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save model and checkpoint every 100 iterations
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(infer_similarity_metric.state_dict(), best_model_path)
            # print(f"Best model updated at iteration {i}, Loss: {best_loss:.4f}")

        if i % 100 == 0:
            checkpoint = {
                'iteration': i,
                'model_state_dict': infer_similarity_metric.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, checkpoint_path)
            # print(f"Checkpoint saved at iteration {i}")
        if i % 10 == 0:
            print(f"Iteration {i}/{iterations}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

    # Stop the batch generator
    # patches.stop()
    print(f"Training finished. Best loss: {best_loss:.4f}")
    del infer_similarity_metric
    torch.cuda.empty_cache()


def main():
    # Fix random seed for reproducibility
    np.random.seed(7)
    torch.manual_seed(7)

    # Hyperparameters
    start_iteration = 1
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
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Create dataloader for KITTI training set
    dataset = KITTIDataset(
        osp.join(data_dir, "training"),
        osp.join(data_dir, "training/disp_noc_0"),
    )
    # Load patch provider
    patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))

    # Initialize the network
    infer_similarity_metric = StereoMatchingNetwork()
    # Set to train
    infer_similarity_metric.train()
    infer_similarity_metric.cuda()
    # uncomment if you don't have a gpu
    # infer_similarity_metric.to('cpu')
    optimizer = torch.optim.SGD(
        infer_similarity_metric.parameters(), lr=learning_rate, momentum=0.9
    )

    # Check if checkpoint exists
    checkpoint_path = osp.join(out_dir, f"iteration_{training_iterations}", "checkpoint.pth")
    if osp.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        infer_similarity_metric.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        print(f"Resumed training from iteration {start_iteration}")
    else:
        start_iteration = 1

    # Start training loop
    training_loop(
        infer_similarity_metric,
        patches,
        optimizer,
        out_dir,
        start_iteration=start_iteration,
        iterations=training_iterations,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
