import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# file_name = "models/z_path_nr_frames_32_epochs_10.pt"
# file_name = "models/z_path_nr_frames_24_handpicked.pt"
file_name = "models/z_path_nr_frames_32_epochs_0.pt"
path = pickle.load(open(file_name, "rb"))

fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot(path[:,0], path[:,1], path[:,4])
ax.plot(path[:,0], path[:,1], path[:,5])
ax.plot(path[:,0], path[:,1], path[:,6])

fig2 = plt.figure(2)
ax = fig2.gca(projection='3d')
ax.plot(path[:,3], path[:,2], path[:,4])
ax.plot(path[:,3], path[:,2], path[:,5])
ax.plot(path[:,3], path[:,2], path[:,6])

fig3 = plt.figure(3)
ax = fig3.gca(projection='3d')
ax.plot(path[:,1], path[:,7], path[:,4])
ax.plot(path[:,1], path[:,7], path[:,5])
ax.plot(path[:,1], path[:,7], path[:,6])


plt.show()
