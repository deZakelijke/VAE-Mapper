import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from VAE_class import VAE


class LossFunctions(object):
    """ Class that contains loss funcions for PathPlanner class

    Each function has as call sign:
    func(decoded_path, decoded_start, decoded_dest, nr_frames)
    Args:
        decoded_path:  Decoded frames in the path for which the loss is calculated.
                       The error is only calculated of these frames
        decoded_start: First decoded frame of total path. Used to calculate 
                       error of next frame
        decoded_dest:  Last decoded frame of total path. Used to calculate
                       error of previous frame
        nr_frames:     Number of frames in the path excluding start and dest
    """

    def l2_loss(decoded_path, decoded_start, decoded_dest, nr_frames):
        """ Pixelwise L2 loss of the frames."""

        loss_tensor = Variable(torch.zeros(nr_frames, 3, 64, 64).double().cuda())
        for i in range(nr_frames): 
            if i is 0:
                loss_tensor[i] = (decoded_start - decoded_path[i]) ** 2 \
                               + (decoded_path[i] - decoded_path[i + 1]) ** 2
            elif i is nr_frames - 1:
                loss_tensor[i] = (decoded_path[i] - decoded_dest) ** 2 \
                               + (decoded_path[i - 1] - decoded_path[i]) ** 2
            else:
                loss_tensor[i] = (decoded_path[i] - decoded_path[i + 1]) ** 2 \
                               + (decoded_path[i - 1] - decoded_path[i]) ** 2

        return torch.sum(loss_tensor)


    def l1_loss(decoded_path, decoded_start, decoded_dest, nr_frames):
        """ Pixelwise L1 loss of the frames."""

        loss_tensor = Variable(torch.zeros(nr_frames, 3, 64, 64).double().cuda())
        for i in range(nr_frames): 
            if i is 0:
                loss_tensor[i] = torch.abs(decoded_start - decoded_path[i]) \
                               + torch.abs(decoded_path[i] - decoded_path[i + 1])
            elif i is nr_frames - 1:
                loss_tensor[i] = torch.abs(decoded_path[i] - decoded_dest) \
                               + torch.abs(decoded_path[i - 1] - decoded_path[i])
            else:
                loss_tensor[i] = torch.abs(decoded_path[i] - decoded_path[i + 1]) \
                               + torch.abs(decoded_path[i - 1] - decoded_path[i])

        return torch.sum(loss_tensor)

