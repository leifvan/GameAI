from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from p17_som_2 import batch_train_som, plot_data_and_som, dist

# ------------------------------------------
# ----------------- TASK 1 -----------------
# ------------------------------------------

path = np.loadtxt("../data/q3dm1-path2.csv", delimiter=',')
weights = batch_train_som(path, k=24, t_max=1000, dist_fn=partial(dist, k2=12),
                          plot_fn=lambda *args: None, plot_every=np.inf)

plot_data_and_som(path, weights, None, None, None, partial(dist, k2=12))
ax = plt.gca()
ax.view_init(elev=45, azim=-135)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.show()

# som error

distances = cdist(path, weights)
shortest = np.min(distances, axis=1)**2
error = shortest.mean()

print("som error")
print(np.round(error, 2))

# ------------------------------------------
# ----------------- TASK 2 -----------------
# ------------------------------------------

v = path[1:] - path[:-1]
mean_v = v.mean(axis=0)
print(np.round(mean_v, 4))

a, _ = kmeans2(v, k=24, iter=100, minit='++')
mean_a = a.mean(axis=0)
print(np.round(mean_a, 4))

