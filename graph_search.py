import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from VAE_class import VAE
from VideoData import VideoData
from torchvision.utils import save_image
# load all data and encode it, add it to tensor
# make list (tensor?) of visited, maybe list of indices
# for vertex in list_of_unvisited_vertices:
#   for edge in vertex:
#       if edge.dest in visited:
#           if edge.len < curr_edge_to_dest.len:
#               make edge curr_edge_to_dest
#       else:
#           add edge.dest to visited
#           add edge.dest to todo

class GraphSearcher(object):

    def __init__(self, model_path, nr_images):
        self.model = torch.load(model_path)
        self.model.eval()
        self.nr_images = nr_images
        #self.vertices = Variable(torch.Tensor([])).double().cuda()
        self.vertices = torch.cuda.DoubleTensor([])
        data_loader = DataLoader(VideoData(nr_images), batch_size=1)
        for i, data in enumerate(data_loader):
            print('Loading vertices: {:.1f}%'.format(i/nr_images*100), end='\r')
            data = Variable(data).cuda()
            point = self.model.reparametrize(*self.model.encode(data))
            point = torch.cuda.DoubleTensor(point.data)
            self.vertices = torch.cat([self.vertices, point])
        print('Vertices loaded          ')


    def make_adjacency_matrix(self):
        size = self.nr_images
        self.adjacency_matrix = torch.zeros(size, size)
        for i in range(size):
            print("Calculating adjacenct matrix: {:.1f}%".format(i/size*100), end='\r')
            for j in range(i + 1, size):
                dist = torch.sum((self.vertices[i] - self.vertices[j]) ** 2)
                self.adjacency_matrix[i, j] = dist
                self.adjacency_matrix[j, i] = dist
        print("Adjacency matrix calculated               ")


    def find_path(self, point_a, point_b):
        paths = []
#       Each path is a list containing a list of edges and the length of the last edge
        for i in range(self.nr_images):
            paths.append([[point_a], self.adjacency_matrix[i, point_a]])

# Change saved value to length of longest edge

        for i in range(self.nr_images):
            print("Vertices visited: {:.1f}%".format(i/self.nr_images*100), end='\r')
            for j in range(i + 1, self.nr_images):
                if (paths[j][1] > self.adjacency_matrix[i, j] and 
                   paths[j][1] > paths[i][1]):
                    paths[j][0] = paths[i][0][:]
                    paths[j][0].append(i)
                    if paths[i][1] > self.adjacency_matrix[i, j]:
                        paths[j][1] = paths[i][1]
                    else:
                        paths[j][1] = self.adjacency_matrix[i, j]

        self.latent_path = paths[point_b][0]
        self.latent_path.append(point_b)
        print("Path in latent space calculated: length is {} points".format(
              len(self.latent_path)))

    def convert_path_to_images(self):
        image_path = torch.zeros(len(self.latent_path), 8)
        image_path = Variable(image_path).double().cuda()
        for i in range(image_path.shape[0]):
            image_path[i] = self.vertices[self.latent_path[i]]

        image_path = self.model.decode(image_path)
        save_image(image_path.data, 'results/path_graph.png', 
                   nrow=int(np.sqrt(image_path.shape[0]) + 1))
        print("Path decoded and saved")
        

if __name__ == '__main__':
    model_path = 'models/model_learning-rate_0.001_batch-size_64_epoch_300_nr-images_2000.pt'
    nr_images = 2000
    graph = GraphSearcher(model_path, nr_images)
    a = 255
    b = 415
    graph.make_adjacency_matrix()
    graph.find_path(a, b)
    graph.convert_path_to_images()
