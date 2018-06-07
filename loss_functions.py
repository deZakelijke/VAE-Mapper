import numpy as np
import torch
import time
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
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

    def ssim_loss(decoded_path, decoded_start, decoded_dest, nr_frames):
        start = time.time()
        pad = (5, 5, 5, 5)
        shape = (64, 64)
        decoded_path  = F.pad(decoded_path, pad)
        decoded_start = F.pad(decoded_start, pad)
        decoded_dest  = F.pad(decoded_dest, pad)

        loss = 0
        for i in range(nr_frames):
            if not i % 10:
                print("Frame:", i)
            if i is 0:
                loss -= ssim(decoded_start, decoded_path[i], shape)
            else:
                loss -= ssim(decoded_path[i - 1], decoded_path[i], shape)

            if i is nr_frames - 1:
                loss -= ssim(decoded_path[i], decoded_dest, shape)
            else:
                loss -= ssim(decoded_path[i], decoded_path[i + 1], shape)
        print("Time for epoch:", time.time() - start)
        return loss


def ssim(image_a, image_b, shape):
    image_a = image_a.view(3, 74, 74)
    image_b = image_b.view(3, 74, 74)
    C1 = 1e-10
    C2 = 1e-10
    C3 = 1e-10
    ssim = 0
    for i in range(5, shape[0] + 5):
        for j in range(5, shape[1] + 5):
            m_a = image_a[:, i - 5:i + 5, j - 5:j + 5].mean()
            m_b = image_b[:, i - 5:i + 5, j - 5:j + 5].mean()
            v_a = image_a[:, i - 5:i + 5, j - 5:j + 5] - m_a
            v_b = image_b[:, i - 5:i + 5, j - 5:j + 5] - m_b
            s_a = torch.sqrt(torch.sum(v_a ** 2))
            s_b = torch.sqrt(torch.sum(v_b ** 2))

            I = (2 * m_a * m_b + C1) / (m_a ** 2 + m_b ** 2 + C1)
            C = (2 * s_a * s_b + C2) / (s_a ** 2 + s_b ** 2 + C2)
            S = (torch.sum(v_a * v_b) + C3) / (s_a * s_b + C3)
            ssim += I * C * S
    return ssim
                
