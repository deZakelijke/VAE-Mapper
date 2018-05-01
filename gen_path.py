# Path planning in the latent space

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VideoData import VideoData
from VAE_class import VAE

# Load a trained model
# Get two images
# map to latent space
# Get difference vector
# Should the path itself be a class or the path planning?
# The path consists of a sequence of points in the latent space
# or images, if they have been converted to images
# Path planning is a bit more than that, also contains the planning
# methods


class PathPlanner(object):
    '''
    Class for planning a path through latent space
    '''

    def __init(self, model_path):
        self.model = torch.load(model_path)
        # self.model.eval() ?


    def generate_latent_path(self, image_a, image_b, path_frames):
        '''
        image_a
        image_b
        path_frames
        '''
        #path to image or image as tensor?
        # image_a = tensor?

        z_a = self.model.reparametrize(self.encode(image_a)) 
        z_b = self.model.reparametrize(self.encode(image_b)) 

        # Linear path
        z_diff  = z_b - z_a
        size = (path_frames, 3, 64, 64) 
        z_path = torch.Tensor.new_empty(size)
        for i in range(1, path_frames):
            z_path[i] = z_a + 1 / i * z_diff
        return z_path


    def convert_path_to_images(self, z_path, path_frames):
        image_path = decode(z_path)
        save_image(image_path, 'results/path.png', rows=path_frames + 1)
