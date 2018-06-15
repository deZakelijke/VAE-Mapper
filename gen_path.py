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
from loss_functions import LossFunctions as LF

class IllegalArgumentError(ValueError):
    pass

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

        Generate a linear path in the latent space by encoding
        image_a and image_b. Then creaate a tensor with evenly
        spacced points in the latent space between those two points.

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

    def gradient_descend_path(self, loss_func):
        """ Perform gradient descend on the latent path """

        optimizer.zero_grad()
        loss = loss_func(self.model, self.z_path, self.z_a, self.z_b, self.nr_frames)
        loss.backward(retain_graph=True)
        print('Loss on path: {}'.format(loss.data[0]))
        optimizer.step()


    def simple_path(self, start, dest, nr_frames):
        """ Make simple path from the dataset

        A sequence of evenly spaced images are picked from
        the data set and passed through the model to see if 
        the model could theoretically produce a valid path.
        Prints the error of the path according to the loss function.
        """
            
        stride = (dest - start) // nr_frames
        self.z_path = torch.zeros(self.size).double().cuda()
        self.z_path = Variable(self.z_path)
        self.z_path.retain_grad()

        z_a = Image.open('images/' + str(start) + '.png').resize((64, 64))
        z_a = transforms.ToTensor()(z_a).double().cuda()
        z_a = Variable(z_a)
        z_a = self.model.encode(z_a.view(self.dims))
        self.z_a = self.model.reparametrize(*z_a)

        z_b = Image.open('images/' + str(dest) + '.png').resize((64, 64))
        z_b = transforms.ToTensor()(z_b).double().cuda()
        z_b = Variable(z_b)
        z_b = self.model.encode(z_b.view(self.dims))
        self.z_b = self.model.reparametrize(*z_b)

        for i in range(1, nr_frames + 1):
            tmp = Image.open('images/' + str(start + i * stride) + '.png').resize((64, 64))
            tmp = transforms.ToTensor()(tmp).double().cuda()
            tmp = Variable(tmp, volatile = True)
            tmp = self.model.encode(tmp.view(self.dims))
            tmp = self.model.reparametrize(*tmp)
            self.z_path[i - 1] = tmp
        self.z_path = nn.Parameter(self.z_path.data.clone(), requires_grad=True)


    def convert_path_to_images(self, nr_frames, function, epochs):
        """ Converts the path in the latent space to a sequence of images """
        image_path = torch.zeros(self.size[0] + 2, *self.dims[1:])
        image_path = Variable(image_path).double().cuda()
        image_path[0] = self.model.decode(self.z_a)
        image_path[1:-1] = self.model.decode(self.z_path)
        image_path[-1] = self.model.decode(self.z_b)
        save_image(image_path.data, 'results/path_{}_{}.png'.format(function, epochs), 
                   nrow=int(np.sqrt(nr_frames)) + 1)

    def save_path_to_file(self, save_path, function, epochs):
        """ Save latent path to file """
        numpy_path = np.zeros((self.size[0] + 2, self.size[1]))
        numpy_path[1:-1] = self.z_path.data.cpu().numpy()
        numpy_path[0] = self.z_a.data.cpu().numpy()
        numpy_path[-1] = self.z_b.data.cpu().numpy()
        file_name = "{}z_path_nr_frames_{}_function_{}_epochs_{}.pt".format(
                    save_path, 
                    self.nr_frames + 2,
                    function,
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
                        default = 255,
                        metavar = 'N',
                        help = 'Frame index of start point')
    parser.add_argument('--dest',
                        type = int,
                        default = 415,
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
    parser.add_argument('--loss_function',
                        type = str,
                        default = 'l2',
                        metavar = 'S',
                        help = 'Loss function for gradient descend: \
                                l2 | l1 | ssim | correlation | nearest_image | nearest_latent')

    args = parser.parse_args()
    FUNCTION_MAP = {'l2' : LF.l2_loss,
                    'l1' : LF.l1_loss,
                    'ssim' : LF.ssim_loss,
                    'correlation' : LF.correlation_loss,
                    'nearest_image' : LF.l1_nearest_data,
                    'nearest_latent' : LF.l1_nearest_latent}
    
    start = np.random.randint(1, high=1000)
    dest = np.random.randint(1001, high=2000)
    if not args.loss_function:
        raise IllegalArgumentError('Invalid loss function')
    else:
        loss_func = FUNCTION_MAP[args.loss_function]

    path_planner = PathPlanner(args.model_path, args.nr_frames)

    if args.pick_path and args.load_z_path:
        raise IllegalArgumentError('Can\' have both load_z_path and pick_path enabled')
        
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
        for i in range(1, args.epochs + 1):
            print('Epoch:', i)
            path_planner.gradient_descend_path(loss_func)

    if args.no_display_path:
        path_planner.convert_path_to_images(args.nr_frames, args.loss_function, args.epochs)

    if args.save_z_path:
        path_planner.save_path_to_file(args.save_z_path, args.loss_function, args.epochs)


