# VAE class

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

class VAE(nn.Module):
    def __init__(self, latent_dims=8, image_size=(64, 64)):
        # Inherit from parent class
        super().__init__()

        self.latent_dims = latent_dims
        self.img_chns = 3
        self.image_size = image_size
        self.filters = 32
        self.intermediate_dim = 32 * 32
        self.intermediate_flat = 23328 # 27 * 27 * 32
        self.intermediate_dim2 = 64 // 2 - 5

        # Encoding layers for the mean and logvar of the latent space
        self.conv1_m = nn.Conv2d(self.img_chns, self.img_chns, (2, 2))
        self.conv2_m = nn.Conv2d(self.img_chns, self.filters, (2, 2), stride=(2, 2))
        self.conv3_m = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.conv4_m = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.fc1_m    = nn.Linear(self.intermediate_flat, self.intermediate_dim)
        self.fc2_m    = nn.Linear(self.intermediate_dim, self.latent_dims)

        self.conv1_s = nn.Conv2d(self.img_chns, self.img_chns, (2, 2))
        self.conv2_s = nn.Conv2d(self.img_chns, self.filters, (2, 2), stride=(2, 2))
        self.conv3_s = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.conv4_s = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.fc1_s    = nn.Linear(self.intermediate_flat, self.intermediate_dim)
        self.fc2_s    = nn.Linear(self.intermediate_dim, self.latent_dims)


        # Decoding layers
        self.fc1_d    = nn.Linear(self.latent_dims, self.intermediate_dim)
        self.fc2_d    = nn.Linear(self.intermediate_dim, self.filters * self.intermediate_dim2 * self.intermediate_dim2)
        self.deConv1 = nn.ConvTranspose2d(self.filters, self.filters, 3, stride=(1, 1))
        self.deConv2 = nn.ConvTranspose2d(self.filters, self.filters, 3, stride=(1, 1))
        self.deConv3 = nn.ConvTranspose2d(self.filters, self.filters, (3, 3), stride=(2, 2))
        self.deConv4 = nn.ConvTranspose2d(self.filters, self.img_chns, 2)

        # Other network componetns
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        h1_m = self.relu(self.conv1_m(x))
        h2_m = self.relu(self.conv2_m(h1_m))
        h3_m = self.relu(self.conv3_m(h2_m))
        h4_m = self.relu(self.conv4_m(h3_m))
        h5_m = self.relu(self.fc1_m(h4_m.view(-1, self.intermediate_flat)))
        h6_m = self.dropout(h5_m)

        h1_s = self.relu(self.conv1_m(x))
        h2_s = self.relu(self.conv2_m(h1_s))
        h3_s = self.relu(self.conv3_m(h2_s))
        h4_s = self.relu(self.conv4_m(h3_s))
        h5_s = self.relu(self.fc1_m(h4_s.view(-1, self.intermediate_flat)))
        h6_s = self.dropout(h5_s)
        return self.fc2_m(h6_m), self.fc2_s(h6_s)

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
        h1 = self.relu(self.fc1_d(z))
        h2 = self.dropout(h1)
        h3 = self.relu(self.fc2_d(h2))
        h4 = h3.view(-1, self.filters, self.intermediate_dim2, self.intermediate_dim2)
        h5 = self.relu(self.deConv1(h4))
        h6 = self.relu(self.deConv2(h5))
        h7 = self.relu(self.deConv3(h6))
        h8 = self.sigmoid(self.deConv4(h7))
        return h8

    # encode and decode a data point
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3, *self.image_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar



