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


class PathPlanner(nn.Module):
    '''
    Class for planning a path through latent space
    '''

    def __init__(self, model_path, nr_frames):
        super().__init__()

        self.dims = (-1, 3, 64, 64)
        self.size = (nr_frames + 1, 8)
        self.model = torch.load(model_path)


    def generate_latent_path(self, image_a, image_b, nr_frames):
        '''
        image_a
        image_b
        nr_frames
        '''

        image_a = transforms.ToTensor()(image_a).double().cuda()
        image_a = Variable(image_a)
        z_a = self.model.encode(image_a.view(self.dims))
        z_a = self.model.reparametrize(*z_a) 

        image_b = transforms.ToTensor()(image_b).double().cuda()
        image_b = Variable(image_b)
        z_b = self.model.encode(image_b.view(self.dims))
        z_b = self.model.reparametrize(*z_b)

        # Linear path
        z_diff  = z_b - z_a
        self.z_path = torch.zeros(self.size).double().cuda()
        self.z_path = Variable(self.z_path)
        self.z_path[-1] = z_b
        for i in range(0, nr_frames + 1):
            tmp = z_a + i / (nr_frames + 1) * z_diff
            self.z_path[i] = tmp

    def gradient_descend_path(self):
        optimizer.zero_grad()
        decoded_path  = self.model.decode(self.z_path[1:-1])
        decoded_start = self.model.decode(self.z_path[0])
        decoded_dest  = self.model.decode(self.z_path[-1])
        loss = self.loss_function(decoded_path, decoded_start, decoded_dest, nr_frames)
        loss.backward(decoded_path, retain_graph=True)
        print('Loss on path: {}'.format(loss.data[0]))
        optimizer.step()

    def loss_function(self, decoded_path, decoded_start, decoded_dest, nr_frames):
        loss_tensor = Variable(torch.zeros(nr_frames - 1, 3, 64, 64).double().cuda())
        for i in range(nr_frames - 1): 
            if i is 0:
                loss_tensor[i] = (decoded_start - decoded_path[i]) ** 2
            elif i is nr_frames - 2:
                loss_tensor[i] = (decoded_path[i] - decoded_dest) ** 2
            else:
                loss_tensor[i] = (decoded_path[i] - decoded_path[i + 1]) ** 2
        return torch.sum(loss_tensor)

    def simple_path(self, start, dest, nr_frames):
        stride = (dest - start) // nr_frames
        self.z_path = torch.zeros(self.size).double().cuda()
        self.z_path = Variable(self.z_path, volatile = True)
        for i in range(0, nr_frames + 1):
            tmp = Image.open('images/' + str(start + i * stride) + '.png').resize((64, 64))
            tmp = transforms.ToTensor()(tmp).double().cuda()
            tmp = Variable(tmp, volatile = True)
            tmp = self.model.encode(tmp.view(self.dims))
            tmp = self.model.reparametrize(*tmp)
            self.z_path[i] = tmp

    def convert_path_to_images(self, nr_frames):
        '''
        nr_frames
        '''
        image_path = self.model.decode(self.z_path)
        save_image(image_path.data, 'results/path.png', nrow=nr_frames + 1)


if __name__ == '__main__':
    model_path = 'models/model_learning-rate_0.001_batch-size_128_epoch_{0}_nr-images_2000.pt'.format(
                 300)
    nr_frames = 15

    
    start = np.random.randint(1, high=1000)
    start = 255
    dest = np.random.randint(1001, high=2000)
    dest = 415
    simple_path = False
    learning_rate = 1e-3
    epochs = 2

    path_planner = PathPlanner(model_path, nr_frames)

    if simple_path:
        path_planner.simple_path(start, dest, nr_frames)
    else:
        start = Image.open('images/' + str(start) + '.png').resize((64, 64))
        dest = Image.open('images/' + str(dest) + '.png').resize((64, 64))
        path_planner.generate_latent_path(start, dest, nr_frames)
        optimizer = optim.Adam([path_planner.z_path[1:-1]], lr = learning_rate)
        for i in range(epochs):
            print('Epoch:', i)
            path_planner.gradient_descend_path()

    path_planner.convert_path_to_images(nr_frames)


