from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VideoData import VideoData


# argparsing
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
                    # was --no-cude in original
parser.add_argument('--cuda', action='store_true', default=False,
                            help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# get MNIST data
# TODO replace with own data
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#train = datasets.MNIST('../data', train=True, download=True,
#            transform=transforms.ToTensor(),
#            batch_size=args.batch_size, shuffle=True, **kwargs)
#test = datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#            batch_size=args.batch_size, shuffle=True, **kwargs

dataset = VideoData()

train_loader = torch.utils.data.DataLoader(dataset)
test_loader = torch.utils.data.DataLoader(dataset)


class VAE(nn.Module):
    def __init__(self):
        # Inherit from parent class
        super().__init__()

        # defines layers? 
        #first number is 28*28, so size of input, same as last
        # first three are used in encoding, last two in decoding
        # Does that mean Z is 20d?
        # input_size = 784
        self.flat_input_size = 2764800 # 1280 * 720 RGB
        self.fc1 = nn.Linear(self.flat_input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    # Read reparametrization trick again
    # Seems to draw from normal dist with std 0.5 and mu=mu
    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))


    # encode and decode a data point
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.flat_input_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar



model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)


# Total loss function
def loss_function(recon_x, x, mu, logvar):
    size = 2764800
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, size), size_average = False)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # reparametrization?
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    # model is a VAE object
    model.train()
    train_loss = 0
    # train_loader is from torch.utils.?.mnist
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        # What is logvar and how is the object used as a function?
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        # backprop?
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        # Print intermediate results
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
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile = True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            size = (3, 1280, 720)
            comparison = torch.cat([data[:n], 
                                   recon_batch.view(args.batch_size, *size)[:n]])
            save_image(comparison.data.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow = n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, 20))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    size = (3, 1280, 720)
    save_image(sample.data.view(64, *size), 
               'results/sample_' + str(epoch) + '.png')
