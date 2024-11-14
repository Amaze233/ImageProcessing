import numpy as np
import torch


class StereoMatchingNetwork(torch.nn.Module):
    """
    The network should consist of the following layers:
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - functional.normalize(..., dim=1, p=2)

    Remark: Note that the convolutional layers expect the data to have shape
        `batch size * channels * height * width`. Permute the input dimensions
        accordingly for the convolutions and remember to revert it before returning the features.
    """

    def __init__(self):
        """
        Implementation of the network architecture.
        Layer output tensor size: (batch_size, n_features, height - 8, width - 8)
        """

        super().__init__()
        gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = gpu

        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 5)
        # -------------------------------------
        # Define the network architecture here
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)

    def forward(self, X):
        """
        The forward pass of the network. Returns the features for a given image patch.

        Args:
            X (torch.Tensor): image patch of shape (batch_size, height, width, n_channels)

        Returns:
            features (torch.Tensor): predicted normalized features of the input image patch X,
                               shape (batch_size, height - 8, width - 8, n_features)
        """

        #######################################
        # -------------------------------------
        # TODO: ENTER CODE HERE (EXERCISE 5)
        # -------------------------------------
        # Implement the forward pass here
        X = X.permute(0, 3, 1, 2).cuda()
        # X = X.permute(0, 3, 1, 2)
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.conv3(X)
        X = self.relu3(X)
        X = self.conv4(X)

        # Normalize
        X = torch.nn.functional.normalize(X, dim=1, p=2)
        X = X.permute(0, 2, 3, 1).cuda()
        return X


def calculate_similarity_score(infer_similarity_metric, Xl, Xr):
    """
    Computes the similarity score for two stereo image patches.

    Args:
        infer_similarity_metric (torch.nn.Module):  pytorch module object
        Xl (torch.Tensor): tensor holding the left image patch
        Xr (torch.Tensor): tensor holding the right image patch

    Returns:
        score (torch.Tensor): the similarity score of both image patches which is the dot product of their features
    """

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5)
    # -------------------------------------
    features_left = infer_similarity_metric(Xl) # Shape: (batch_size, height - 8, width - 8, n_features)
    features_right = infer_similarity_metric(Xr)

    # 计算余弦相似度 (点积)- score = features_left * features_right / ||features_left||*||features_right||
    # 由于已经对提取的特征值进行过归一化,所以可以直接计算点积
    cosine_similarity = torch.sum(features_left * features_right, dim=-1)
    return cosine_similarity # (batch_size, height - 8, width - 8)

