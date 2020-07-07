from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.vq
from mpl_toolkits.mplot3d import Axes3D

k = 10
data = np.loadtxt("../data/q3dm1-path2.csv", delimiter=',')
centroids, indices = scipy.cluster.vq.kmeans2(data, k=k, iter=100, minit='++')

fig = plt.figure(figsize=(5,5))
ax = Axes3D(fig)
ax.view_init(elev=45, azim=45)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1, c='black', marker='o')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='o', alpha=1)
plt.show(block=False)

p = np.zeros((k, k))

counter = Counter(zip(indices[:-1], indices[1:]))
for idx, val in counter.items():
    p[idx] = val / len(indices)

np.set_printoptions(linewidth=300)
print(np.round(p, 2))


plt.matshow(p)
plt.show()