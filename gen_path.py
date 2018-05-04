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


class PathPlanner(object):
    '''
    Class for planning a path through latent space
    '''

    def __init__(self, model_path):
        self.model = torch.load(model_path)


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
        size = (nr_frames + 1, 8, 3, 3) 
        self.z_path = torch.zeros(size).double().cuda()
        self.z_path = Variable(self.z_path, volatile = True)
        self.z_path[-1] = z_b
        for i in range(0, nr_frames + 1):
            tmp = z_a + i / (nr_frames + 1) * z_diff
            self.z_path[i] = tmp

        # Improve linear path with gradient descend


    def convert_path_to_images(self, nr_frames):
        '''
        nr_frames
        '''
        image_path = self.model.decode(self.z_path)
        save_image(image_path.data, 'results/path.png', nrow=nr_frames + 1)


if __name__ == '__main__':
    model_path = 'models/model_learning-rate_0.001_batch-size_64_epoch_{0}_nr-images_2000.pt'.format(
                 50)
    nr_frames = 10

    start = np.random.randint(1, high=1000)
    start = Image.open('images/' + str(start) + '.png').resize((64, 64))
    dest = np.random.randint(1001, high=2000)
    dest = Image.open('images/' + str(dest) + '.png').resize((64, 64))

    path_planner = PathPlanner(model_path)
    path_planner.generate_latent_path(start, dest, nr_frames)
    path_planner.convert_path_to_images(nr_frames)
