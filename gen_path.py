# Path planning in the latent space

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from PIL import Image
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

    def __init__(self, model_path):
        self.model = torch.load(model_path)
        # self.model.eval() ?


    def generate_latent_path(self, image_a, image_b, nr_frames):
        '''
        image_a
        image_b
        nr_frames
        '''
        dims = (-1, 3, 64, 64)

        image_a = transforms.ToTensor()(image_a).double().cuda()
        image_a = Variable(image_a, volatile = True)
        z_a = self.model.encode(image_a.view(dims))
        z_a = self.model.reparametrize(*z_a) 

        image_b = transforms.ToTensor()(image_b).double().cuda()
        image_b = Variable(image_b, volatile = True)
        z_b = self.model.encode(image_b.view(dims))
        z_b = self.model.reparametrize(*z_b)

        # Linear path
        z_diff  = z_b - z_a
        size = (nr_frames, 8, 3, 3) 
        self.z_path = torch.zeros(size).double().cuda()
        self.z_path = Variable(self.z_path, volatile = True)
        for i in range(1, nr_frames):
            tmp = z_a + 1 / i * z_diff
            self.z_path[i] = tmp


    def convert_path_to_images(self, nr_frames):
        '''
        nr_frames
        '''
        image_path = self.model.decode(self.z_path)
        save_image(image_path.data, 'results/path.png', nrow=nr_frames + 1)


if __name__ == '__main__':
    model_path = 'models/model_learning-rate_0.01_batch-size_128_epoch_1.pt'
    nr_frames = 10

    start = np.random.randint(1, high=1000)
    start = Image.open('images/' + str(start) + '.png').resize((64, 64))
    dest = np.random.randint(1001, high=2000)
    dest = Image.open('images/' + str(dest) + '.png').resize((64, 64))

    path_planner = PathPlanner(model_path)
    path_planner.generate_latent_path(start, dest, nr_frames)
    path_planner.convert_path_to_images(nr_frames)
