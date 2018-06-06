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
from VAE_with_Disc import VAE, Discriminator


def train(epoch):
    VAE_model.train()
    GAN_model.train()

    train_loss = [0, 0]
    for batch_idx, data in enumerate(train_loader):
        labels = torch.zeros(data.shape[0], 1)
        labels = Variable(labels).double()
        noise_variable = torch.zeros(data.shape[0], latent_dims).double().normal_()
        noise_variable = Variable(noise_variable).double()
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
            noise_variable = noise_variable.cuda()


        # Optimize discriminator
        GAN_model.zero_grad()
        labels.fill_(1)
        #noise_variable.resize_(data.size(0), latent_dims, 1, 1)

        predicted_real_labels  = GAN_model(data)
        real_GAN_loss = GAN_model.loss_function(predicted_real_labels, labels)
        real_GAN_loss.backward()
        
        gen_data = VAE_model.decode(noise_variable)
        labels.fill_(0)
        predicted_fake_labels = GAN_model(gen_data.detach())
        fake_GAN_loss = GAN_model.loss_function(predicted_fake_labels, labels)
        fake_GAN_loss.backward()
        GAN_opt.step()
        
        GAN_loss = real_GAN_loss.data[0] + fake_GAN_loss.data[0]
        train_loss[0] += GAN_loss

        # Optimize VAE
        VAE_model.zero_grad()
        recon_batch, mu, logvar = VAE_model(data)
        
        labels.fill_(1)
        predicted_gen_labels = GAN_model.discriminate(recon_batch)
        rec_loss, gen_loss = VAE_model.loss_function(recon_batch, data, mu, 
                                                     logvar, predicted_gen_labels, labels)
        rec_loss.backward(retain_graph = True)
        gen_loss.backward()
        VAE_opt.step()
                
        VAE_loss = rec_loss.data[0] + gen_loss.data[0]
        train_loss[1] += VAE_loss


        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%]\nGAN loss: {:.6f}, VAE loss: {:.6f}'.format(
                    epoch, 
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    GAN_loss / len(data),
                    VAE_loss / len(data),
                    ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, (train_loss[0] + train_loss[1]) / len(train_loader.dataset)))


def test(epoch):
    VAE_model.eval()
    GAN_model.eval()

    # Don't test every epoch
    if epoch % 5:
        return

    test_loss = 0
    for i, data in enumerate(test_loader):
        labels = torch.ones(data.shape[0], 1)
        labels = Variable(labels).double()
        data = Variable(data, volatile = True)
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()

        recon_batch, mu, logvar = VAE_model(data)
        disc_recon_data = GAN_model(recon_batch)
        loss = VAE_model.loss_function(recon_batch, data, mu, logvar, disc_recon_data, labels)
        test_loss += loss[0].data[0] + loss[1].data[0]

        if i == 0 and epoch % 50 == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], 
                                   recon_batch.view(args.batch_size, *size)[:n]])
            save_image(comparison.data.cpu(), 'results/reconstruction_with_GAN_' + str(epoch) + '.png', nrow = n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



# argparsing
parser = argparse.ArgumentParser(description='VAE trainer for path planning')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
parser.add_argument('--cuda', action='store_true', default=False,
                            help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
parser.add_argument('--nr-images', type=int, default=2000, metavar='N',
                            help='Number of images from the dataset that are used (defaut: 2000)')
parser.add_argument('--save-path', type=str, default='models/', metavar='P',
                            help='Path to file to save model')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='L',
                            help='The learning rate of the model')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
dataset = VideoData(nr_images=args.nr_images)

train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=args.batch_size, 
                                           shuffle=True,
                                           **kwargs)
test_loader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=args.batch_size, 
                                          shuffle=True,
                                          **kwargs)

latent_dims = 8
image_size = (64, 64)
size = (3, *image_size)
VAE_model = VAE(latent_dims, image_size).double()
GAN_model = Discriminator(latent_dims, image_size).double()
if args.cuda:
    VAE_model.cuda()
    GAN_model.cuda()

VAE_opt = optim.Adam(VAE_model.parameters(), lr = args.learning_rate, betas = (0.5, 0.999))
GAN_opt = optim.Adam(GAN_model.parameters(), lr = args.learning_rate, betas = (0.5, 0.999))



for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch % 50 == 0:
        sample = Variable(torch.randn(64, latent_dims)).double()
        if args.cuda:
            sample = sample.cuda()
        sample = VAE_model.decode(sample).cpu()
        save_image(sample.data.view(64, *size), 
                   'results/sample_VAE_GAN_' + str(epoch) + '.png')

    # Save model
    if args.save_path and not epoch % 50:
        save_file = '{0}VAE_model_learning-rate_{1}_batch-size_{2}_epoch_{3}_nr-images_{4}.pt'.format(
                    args.save_path,
                    args.learning_rate,
                    args.batch_size,
                    epoch,
                    args.nr_images)
        torch.save(VAE_model, save_file)
        save_file = '{0}GAN_model_learning-rate_{1}_batch-size_{2}_epoch_{3}_nr-images_{4}.pt'.format(
                    args.save_path,
                    args.learning_rate,
                    args.batch_size,
                    epoch,
                    args.nr_images)
        torch.save(GAN_model, save_file)
