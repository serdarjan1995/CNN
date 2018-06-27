# -*- coding: utf-8 -*-
"""
Created on Wed May  9 17:10:32 2018

@author: Sardor
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

start = -5.0
end = 5.0
step = 0.5

X = np.arange(start,end,step)
Y = np.arange(start,end,step)
np.random.shuffle(X)
np.random.shuffle(Y)


# for plotting 3d
x, y = np.meshgrid(X, Y)
z = x * y
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,  
                       shade=True,linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.title('Dataset',fontsize=16)
plt.show()
print('\n\n')


Z = X * Y
X = np.array([X, Y])
X = np.transpose(X)


#sigmoid
sig = lambda t: 1/(1+np.exp(-t))
layer_1_w = np.zeros((2,3))
layer_2_w = np.zeros((3,1))

eta = 0.1
eps_err = 0.01

itera = 0
iterMax = 1000

error=1
while(error>eps_err and itera<iterMax):
    for x,z in zip(X, Z):
        x = x[np.newaxis]
        layer_1 = sig(np.dot(x, layer_1_w))   
        layer_2 = sig(np.dot(layer_1, layer_2_w))
        
        layer_2_delta = (layer_2-z)*layer_2*(1-layer_2)
        layer_1_delta = np.dot(layer_2_delta, layer_2_w.T)*(layer_1)*(1-layer_1)
        
        layer_2_w -= np.dot(layer_1.T, layer_2_delta)*eta
        layer_1_w -= np.dot(x.T, layer_1_delta)*eta
    error = 0.5*sum((Z-layer_2)**2)
    error = np.mean(error)
    itera += 1
print('iterations:',itera)

r = np.array([0.1,0.1])
r = r[np.newaxis]
print('Predictions:', r)
layer_1 = sig(np.dot(r, layer_1_w))
layer_2 = sig(np.dot(layer_1, layer_2_w))

print('Result:',layer_2)