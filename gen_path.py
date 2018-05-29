# Path planning in the latent space

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
import pickle
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VideoData import VideoData
from VAE_class import VAE


class PathPlanner(nn.Module):
    """ Class for planning a path through latent space

    Generates a path through a trained latent space. This path can
    be decoded and turned into a sequence of frames, yielding a (video) path
    from the start to the end.

    Args:
        model_path (str): path to a trained VAE model
        nr_frames (int): number of frames excluding the start and end frame

    """

    def __init__(self, model_path, nr_frames):
        super().__init__()

        self.dims = (-1, 3, 64, 64)
        self.size = (nr_frames, 8)
        self.model = torch.load(model_path)
        self.nr_frames = nr_frames


    def generate_latent_path(self, image_a, image_b, nr_frames):
        """ Generate a linear interpolation in the latent space

        Args:
            image_a: start of interpolation, PIL image
            image_b: Destination og interpolation, PIL image
            nr_frames: Number of frames that are sampled from the  
                       interpolated line
        """

        image_a = transforms.ToTensor()(image_a).double().cuda()
        image_a = Variable(image_a)
        z_a = self.model.encode(image_a.view(self.dims))
        self.z_a = self.model.reparametrize(*z_a) 

        image_b = transforms.ToTensor()(image_b).double().cuda()
        image_b = Variable(image_b)
        z_b = self.model.encode(image_b.view(self.dims))
        self.z_b = self.model.reparametrize(*z_b)

        # Linear path
        z_diff  = self.z_b - self.z_a
        self.z_path = torch.zeros(self.size).double().cuda()
        self.z_path = Variable(self.z_path)
        self.z_path.retain_grad()
        for i in range(1, nr_frames + 1):
            tmp = self.z_a + i / (nr_frames + 1) * z_diff
            self.z_path[i - 1] = tmp
        self.z_path = nn.Parameter(self.z_path.data.clone(), requires_grad=True)

    def gradient_descend_path(self):
        """ Perform gradient descend on the latent path """

        optimizer.zero_grad()
        decoded_path  = self.model.decode(self.z_path)
        decoded_start = self.model.decode(self.z_a)
        decoded_dest  = self.model.decode(self.z_b)
        loss = self.loss_function(decoded_path, decoded_start, decoded_dest, self.nr_frames)
        loss.backward(decoded_path, retain_graph=True)
        print('Loss on path: {}'.format(loss.data[0]))
        optimizer.step()

    def loss_function(self, decoded_path, decoded_start, decoded_dest, nr_frames):
        """ Returns loss as ssd between previous and next frame for all frames

        Args:
            decoded_path:  Decoded frames in the path for which the loss is calculated.
                           The error is only calculated of these frames
            decoded_start: First decoded frame of total path. Used to calculate 
                           error of next frame
            decoded_dest:  Last decoded frame of total path. Used to calculate
                           error of previous frame
            nr_frames:     Number of frames in the path excluding start and dest
        """

        # Find affine(?) matrix A and B that are the best affine map for both
        # g(z_i) -> g(z_i-1) and g(z_i) -> g(z_i+1) respectively
        # Options:
        #   Init A and B random, use GD 

        #loss_tensor = Variable(torch.zeros(nr_frames, 3, 64, 64).double().cuda())
        loss_tensor = 0
        for i in range(nr_frames): 
            if i is 0:
                loss_tensor[i] = (decoded_start - decoded_path[i]) ** 2 \
                               + (decoded_path[i] - decoded_path[i + 1]) ** 2
            elif i is nr_frames - 1:
                loss_tensor[i] = (decoded_path[i] - decoded_dest) ** 2 \
                               + (decoded_path[i - 1] - decoded_path[i]) ** 2
            else:
                loss_tensor[i] = (decoded_path[i] - decoded_path[i + 1]) ** 2 \
                               + (decoded_path[i - 1] - decoded_path[i]) ** 2


            if i is 0:
                # find Affine_prev with start
                # Get first loss part
            else:
                # find Affine_prev with i-1
                # Get first loss part

            if i is nr_frames - 1:
                # find Affine_next wit hend
                # Get second loss part 
            else:
                # Find Affine_next with i + 1
                # get second part of loss
        return torch.sum(loss_tensor)

    def simple_path(self, start, dest, nr_frames):
        """ Make simple path from the dataset

            A sequence of evenly spaced images are picked from
            the data set and passed through the model to see if 
            the model could theoretically produce a valid path.
        """
            
        stride = (dest - start) // nr_frames
        self.z_path = torch.zeros(self.size).double().cuda()
        self.z_path = Variable(self.z_path, volatile = True)

        z_a = Image.open('images/' + str(start) + '.png').resize((64, 64))
        z_a = transforms.ToTensor()(z_a).double().cuda()
        z_a = Variable(z_a, volatile = True)
        z_a = self.model.encode(z_a.view(self.dims))
        self.z_a = self.model.reparametrize(*z_a)
        z_b = Image.open('images/' + str(dest) + '.png').resize((64, 64))
        z_b = transforms.ToTensor()(z_b).double().cuda()
        z_b = Variable(z_b, volatile = True)
        z_b = self.model.encode(z_b.view(self.dims))
        self.z_b = self.model.reparametrize(*z_b)

        for i in range(1, nr_frames + 1):
            tmp = Image.open('images/' + str(start + i * stride) + '.png').resize((64, 64))
            tmp = transforms.ToTensor()(tmp).double().cuda()
            tmp = Variable(tmp, volatile = True)
            tmp = self.model.encode(tmp.view(self.dims))
            tmp = self.model.reparametrize(*tmp)
            self.z_path[i - 1] = tmp

    def convert_path_to_images(self, nr_frames):
        """ Converts the path in the latent space to a sequence of images """
        image_path = torch.zeros(self.size[0] + 2, *self.dims[1:])
        image_path = Variable(image_path).double().cuda()
        image_path[0] = self.model.decode(self.z_a)
        image_path[1:-1] = self.model.decode(self.z_path)
        image_path[-1] = self.model.decode(self.z_b)
        save_image(image_path.data, 'results/path.png', nrow=int(np.sqrt(nr_frames)) + 1)

    def save_path_to_file(self, save_path, epochs):
        """ Save latent path to file """
        numpy_path = np.zeros((self.size[0] + 2, self.size[1]))
        numpy_path[1:-1] = self.z_path.data.cpu().numpy()
        numpy_path[0] = self.z_a.data.cpu().numpy()
        numpy_path[-1] = self.z_b.data.cpu().numpy()
        file_name = "{0}z_path_nr_frames_{1}_epochs_{2}.pt".format(
                    save_path, 
                    self.nr_frames + 2,
                    epochs)
        pickle.dump(numpy_path, open(file_name, 'wb'))

    def load_path_from_file(self, load_path):
        """ Load latent path from file 
       
        Args:
            load_path: filename of the pickle file you want to load

        Returns:
            returns the number of frames from the loaded z_path
        """
        numpy_path = pickle.load(open(load_path, "rb"))
        self.z_a = Variable(torch.from_numpy(numpy_path[0]).double().cuda())
        self.z_b = Variable(torch.from_numpy(numpy_path[-1]).double().cuda())
        self.z_path = nn.Parameter(torch.from_numpy(numpy_path[1:-1]).double().cuda(), requires_grad=True)
        self.nr_frames = len(numpy_path[1:-1])
        self.size = (self.nr_frames, self.size[1])
        return self.nr_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path generator through a learned latent space')
    parser.add_argument('--model_path', 
                        type = str, 
                        default = 'models/model_learning-rate_0.001_' + \
                        'batch-size_64_epoch_{0}_nr-images_2000.pt'.format(300),
                        metavar = 'P', 
                        help = 'Path to the trained VAE model')
    parser.add_argument('--nr_frames', 
                        type = int,
                        default = 22,
                        metavar = 'N',
                        help = 'Number of frames in the output excluding start and end frame (default: 22)')
    parser.add_argument('--start',
                        type = int,
                        default = 1,
                        metavar = 'N',
                        help = 'Frame index of start point')
    parser.add_argument('--dest',
                        type = int,
                        default = 60,
                        metavar = 'N',
                        help = 'Frame index of destination point')
    parser.add_argument('--no_display_path',
                        action = 'store_false',
                        default = True,
                        help = 'Don\'t save path to sequence of images')
    parser.add_argument('--save_z_path',
                        type = str,
                        default = 'models/',
                        metavar = 'P',
                        help = 'Path to folder to store latent z_path in (default: models/)')
    parser.add_argument('--load_z_path',
                        type = str,
                        default = None,
                        metavar = 'P',
                        help = 'Path to pickle file to load latent z_path with')
    parser.add_argument('--pick_path',
                        action = 'store_true',
                        default = False,
                        help = 'Create path by picking images from the  data instead of generating it')
    parser.add_argument('--epochs',
                        type = int,
                        default = 10,
                        metavar = 'N',
                        help = 'Number of epochs in gradient descend')
    parser.add_argument('--learning_rate',
                        type = float,
                        default = 1e-2,
                        metavar = 'L',
                        help = 'Learning rate of gradient descend')

    args = parser.parse_args()
    
    start = np.random.randint(1, high=1000)
    dest = np.random.randint(1001, high=2000)

    path_planner = PathPlanner(args.model_path, args.nr_frames)

    if args.pick_path and args.load_z_path:
        raise RuntimeError('Can\' have both load_z_path and pick_path enabled')
        
    if args.pick_path:
        path_planner.simple_path(args.start, args.dest, args.nr_frames)
    elif args.load_z_path:
        args.nr_frames = path_planner.load_path_from_file(args.load_z_path)
    else:
        start = Image.open('images/' + str(args.start) + '.png').resize((64, 64))
        dest = Image.open('images/' + str(args.dest) + '.png').resize((64, 64))
        path_planner.generate_latent_path(start, dest, args.nr_frames)

    if args.epochs:
        optimizer = optim.Adam([path_planner.z_path], lr = args.learning_rate)
        for i in range(args.epochs):
            print('Epoch:', i)
            path_planner.gradient_descend_path()

    if args.no_display_path:
        path_planner.convert_path_to_images(args.nr_frames)

    if args.save_z_path:
        path_planner.save_path_to_file(args.save_z_path, args.epochs)


