# VAE class

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

class VAE(nn.Module):
    def __init__(self):
        # Inherit from parent class
        super().__init__()

        self.flat_input_size = 6912 # 64 * 36 * 3
        self.latent_dims = 8

        # Encoding layers
        self.conv1_m = nn.Conv2d(3, 16, 3, stride=3, padding=1)
        self.conv2_m = nn.Conv2d(16, self.latent_dims, 3, stride=2, padding=1)
        self.conv1_s = nn.Conv2d(3, 16, 3, stride=3, padding=1)
        self.conv2_s = nn.Conv2d(16, self.latent_dims, 3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(2)

        # Decoding layers
        self.deConv1 = nn.ConvTranspose2d(self.latent_dims, 16, 5, stride=3)
        self.deConv2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)
        self.deConv3 = nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        h1_m = self.maxpool(self.relu(self.conv1_m(x)))
        h2_m = self.maxpool(self.relu(self.conv2_m(h1_m)))
        h1_s = self.maxpool(self.relu(self.conv1_s(x)))
        h2_s = self.maxpool(self.relu(self.conv2_s(h1_s)))
        return h2_m, h2_s

    # Read reparametrization trick again
    # Seems to draw from normal dist with std 0.5 and mu=mu
    def reparametrize(self, mu, logvar):
        if self.training:
            # Shouldnt the dist be drawn from N(0,1)?
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode(self, z):
        h3 = self.relu(self.deConv1(z))
        h4 = self.relu(self.deConv2(h3))
        h5 = self.deConv3(h4)
        return self.sigmoid(h5)


    # encode and decode a data point
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3, 64, 64))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar



