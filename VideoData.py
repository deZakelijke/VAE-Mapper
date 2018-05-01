# Dataset class for the video data
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils


class VideoData(Dataset):
    ''' 
    Class to serve the video data to pytorch 
    '''

    def __init__(self, nr_images=1000, root_dir='images/'):
        '''
        nr_images is the size of the image dataset
        root_dir is the path to the folder where the image dataset is stored
        '''
        self.path = root_dir
        if os.path.isfile(root_dir + str(nr_images + 1) + '.png'):
            self.nr_images = nr_images
        else:
            raise IndexError('Image index does not exist in dir, Dataset not created')

    def __len__(self):
        return self.nr_images

    def __getitem__(self, idx):
        path = self.path + str(idx + 1) + '.png'
        img = Image.open(path)
        img = img.resize((64, 64))
        img = transforms.ToTensor()(img).double()
        return img
