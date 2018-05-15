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

# argparsing
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
parser.add_argument('--cuda', action='store_true', default=False,
                            help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
parser.add_argument('--nr-images', type=int, default=1000, metavar='N',
                            help='Number of images from the dataset that are used (defaut: 1000)')
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

#####
# Class used to be here, moved to a separate file
latent_dims = 8
image_size = (64, 64)
size = (3, *image_size)
model = VAE(latent_dims, image_size).double()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)


# Maybe move functions somewhere else as well?
# TODO?

# Total loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average = False)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile = True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0 and epoch % 100 == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], 
                                   recon_batch.view(args.batch_size, *size)[:n]])
            save_image(comparison.data.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow = n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch % 100 == 0:
        sample = Variable(torch.randn(64, latent_dims)).double()
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, *size), 
                   'results/sample_' + str(epoch) + '.png')

    # Save model
    if args.save_path and not epoch % 50:
        save_file = '{0}model_learning-rate_{1}_batch-size_{2}_epoch_{3}_nr-images_{4}.pt'.format(
                    args.save_path,
                    args.learning_rate,
                    args.batch_size,
                    epoch,
                    args.nr_images)
        torch.save(model, save_file)
