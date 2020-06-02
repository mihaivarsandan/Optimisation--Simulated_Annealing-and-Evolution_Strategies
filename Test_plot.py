from Function import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits import mplot3d

F = Common()
fig, ax = F.plot_func_contour()

array=[]
big_array=[]
for j in range(4):
    for i in range(10):
        p= 2*np.random.uniform(-1, 1, 2)
        array.append(p.tolist())
    big_array.append(array)
    array=[]

for arr in big_array:
    points = np.array(arr)
    plt.scatter(points[:, 0], points[:, 1])


plt.show()