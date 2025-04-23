import numpy as np
import matplotlib.pyplot as plt


mag_data_path="data/mag_data.txt"
data=np.loadtxt(mag_data_path)

N=data.shape[0]

mag_x=data[:,0]
mag_y=data[:,1]
mag_z=data[:,2]


fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(projection='3d')

ax.set_aspect('equal')

ax.scatter(mag_x,mag_y,mag_z,s=1)

plt.show()