import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from VAE_class import VAE
from VideoData import VideoData
from mpl_toolkits.mplot3d import Axes3D

# file_name = "models/z_path_nr_frames_32_epochs_10.pt"
# file_name = "models/z_path_nr_frames_24_handpicked.pt"
#file_name = "models/z_path_nr_frames_32_epochs_0.pt"
#path = pickle.load(open(file_name, "rb"))


points = 200
model_path = 'models/model_learning-rate_0.001_batch-size_64_epoch_{0}_nr-images_2000.pt'.format(300)
point_matrix = np.zeros((points, 8))
model = torch.load(model_path)
print('Model loaded')
image_loader = torch.utils.data.DataLoader(VideoData(nr_images=points), 
                                           batch_size=1,
                                           shuffle=False)
for i, data in enumerate(image_loader):
    if not i % 10:
        print(i)
    data = Variable(data).cuda()
    point = model.encode(data)
    point = model.reparametrize(*point)
    point_matrix[i] = np.array(point.data)


fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot(point_matrix[:,0], point_matrix[:,1], point_matrix[:,4])
ax.plot(point_matrix[:,0], point_matrix[:,1], point_matrix[:,5])
ax.plot(point_matrix[:,0], point_matrix[:,1], point_matrix[:,6])

#fig2 = plt.figure(2)
#ax = fig2.gca(projection='3d')
#ax.plot(path[:,3], path[:,2], path[:,4])
#ax.plot(path[:,3], path[:,2], path[:,5])
#ax.plot(path[:,3], path[:,2], path[:,6])
#
#fig3 = plt.figure(3)
#ax = fig3.gca(projection='3d')
#ax.plot(path[:,1], path[:,7], path[:,4])
#ax.plot(path[:,1], path[:,7], path[:,5])
#ax.plot(path[:,1], path[:,7], path[:,6])


plt.show()
