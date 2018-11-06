import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(Softmax, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.model = nn.Linear(3*32*32, n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
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
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
        images = images.view(-1, 3*32*32)
        scores = F.softmax(self.model(images)).clamp(min=0)



        # maxes = torch.max(xs + torch.log(mask), 1, keepdim=True)[0]
        # masked_exp_xs = torch.exp(xs - maxes) * mask
        # normalization_factor = masked_exp_xs.sum(1, keepdim=True)
        # probs = masked_exp_xs / normalization_factor
        # log_probs = (xs - maxes - torch.log(normalization_factor)) * mask
        #
        # self.save_for_backward(probs, mask)
        # return probs, log_probs
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

