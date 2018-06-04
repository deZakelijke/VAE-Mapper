import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

class VAE_GAN(nn.Module):
    """ Class that combines a VAE and a GAN in one generative model

    Learns a latent epresentation for a set of images while simultaneously 
    being able to generate new sample. 
    The discriminator from the GAN model increases the quality of the results

    Args:
        latent_dims (int): number of dimensions in the latent space z
        image_size (int, int): dimensions of the image data, don't change it
    """
    def __init__(self, latent_dims=8, image_size=(64, 64)):
        super().__init__()

        self.latent_dims = latent_dims
        self.img_chns = 3
        self.image_size = image_size
        self.filters = 32
        self.intermediate_dim = 32 * 32
        self.intermediate_flat = 23328 # 27 * 27 * 32
        self.intermediate_dim2 = 64 // 2 - 5
        self.intermediate_dim_disc = 32 * 60 * 60

        # Encoding layers for the mean and logvar of the latent space
        self.conv1_m  = nn.Conv2d(self.img_chns, self.img_chns, (2, 2))
        self.conv2_m  = nn.Conv2d(self.img_chns, self.filters, (2, 2), stride=(2, 2))
        self.conv3_m  = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.conv4_m  = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.fc1_m    = nn.Linear(self.intermediate_flat, self.intermediate_dim)
        self.fc2_m    = nn.Linear(self.intermediate_dim, self.latent_dims)

        self.conv1_s  = nn.Conv2d(self.img_chns, self.img_chns, (2, 2))
        self.conv2_s  = nn.Conv2d(self.img_chns, self.filters, (2, 2), stride=(2, 2))
        self.conv3_s  = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.conv4_s  = nn.Conv2d(self.filters, self.filters, 3, stride=(1, 1))
        self.fc1_s    = nn.Linear(self.intermediate_flat, self.intermediate_dim)
        self.fc2_s    = nn.Linear(self.intermediate_dim, self.latent_dims)

        # Decoding layers
        self.fc1_de   = nn.Linear(self.latent_dims, self.intermediate_dim)
        self.fc2_de   = nn.Linear(self.intermediate_dim, self.filters * self.intermediate_dim2 * self.intermediate_dim2)
        self.deConv1  = nn.ConvTranspose2d(self.filters, self.filters, 3, stride=(1, 1))
        self.deConv2  = nn.ConvTranspose2d(self.filters, self.filters, 3, stride=(1, 1))
        self.deConv3  = nn.ConvTranspose2d(self.filters, self.filters, (3, 3), stride=(2, 2))
        self.deConv4  = nn.ConvTranspose2d(self.filters, self.img_chns, 2)

        # Discriminating layers
        self.conv1_di = nn.Conv2d(self.img_chns, self.filters, (2, 2))
        self.conv2_di = nn.Conv2d(self.filters, self.filters, (2, 2))
        self.conv3_di = nn.Conv2d(self.filters, self.filters, 3)
        self.fc1_di   = nn.Linear(self.intermediate_dim_disc, self.intermediate_dim)
        self.fc2_di   = nn.Linear(self.intermediate_dim, 1)

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

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode(self, z):
        h1 = self.relu(self.fc1_de(z))
        h2 = self.dropout(h1)
        h3 = self.relu(self.fc2_de(h2))
        h4 = h3.view(-1, self.filters, self.intermediate_dim2, self.intermediate_dim2)
        h5 = self.relu(self.deConv1(h4))
        h6 = self.relu(self.deConv2(h5))
        h7 = self.relu(self.deConv3(h6))
        h8 = self.sigmoid(self.deConv4(h7))
        return h8

    def discriminate(self, x):
        h1 = self.relu(self.conv1_di(x))
        h2 = self.relu(self.conv2_di(h1))
        h3 = self.dropout(h2)
        h4 = self.relu(self.conv3_di(h3))
        h5 = self.dropout(h4)
        h6 = h5.view(-1, self.intermediate_dim_disc)
        h7 = self.relu(self.fc1_di(h6))
        h8 = self.sigmoid(self.fc2_di(h7))
        return h8

    def forward(self, x):
        """ Feed forward function of the network

        Calculates the mean and variance of the input data. Sample z from this distribution
        and draw another random normal z. Decode both z's. Discriminate both decoded values
        and the original data x.

        Args:
            x: Input data 
        """
        mu, logvar = self.encode(x.view(-1, 3, *self.image_size))
        z_x = self.reparametrize(mu, logvar)
        z_p = Variable(mu.data.new(mu.size()).normal_())

        recon_x = self.decode(z_x)
        recon_x_p = self.decode(z_p)

        recon_x_disc = self.discriminate(recon_x)
        z_dec_disc = self.discriminate(recon_x_p)
        x_disc = self.discriminate(x)
            
        return recon_x, mu, logvar, recon_x_disc, x_disc, z_dec_disc

    def loss_function(self, recon_x, x, mu, logvar, recon_x_disc, x_disc, z_dec_disc, gamma):
        BCE  = F.binary_cross_entropy(recon_x, x, size_average = False)
        KLD  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        L_enc  = BCE + KLD
        L_dec  = BCE + KLD - torch.sum(torch.log(z_dec_disc)) * gamma
        L_disc = -torch.sum(torch.log(x_disc) + torch.log(1- z_dec_disc))
        return L_enc, L_dec, L_disc

    def discriminator_loss(self, x_disc, z_dec_disc):
        """ Separate loss function to train the discriminator beforehand

        """
        loss = -torch.sum(torch.log(x_disc) + torch.log(1 - z_dec_disc))
        return loss
