import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        # max_n_communities = 2
        d= 100      # dimensionality of the graph embeddings
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################

        # self.model = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     Flatten(),
        #     nn.Linear(5408, 1024),  # 5408=32*13*13 input size
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 10),
        # )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(16384, 1024),  # 16384=64*32*32 input size
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, 10),
        )
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        scores=self.model(images)
        return scores
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        # return F.log_softmax(out, dim=1)

