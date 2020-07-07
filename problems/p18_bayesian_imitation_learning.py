import random
from collections import Counter
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.vq
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D

from p17_som_2 import dist, batch_train_som, plot_data_and_som

path = np.loadtxt("../data/q3dm1-path2.csv", delimiter=',')

# ---------
# task 18.1
# ---------

v = path[1:] - path[:-1]

# ---------
# task 18.2
# ---------

kA = 9
action_primitives, ap_indices = scipy.cluster.vq.kmeans2(v, k=kA, iter=100, minit='++')

# ---------
# task 18.3
# ---------

kS = 20
t_max = 1000
states = batch_train_som(path, k=kS, t_max=t_max, dist_fn=partial(dist, k2=kS // 2),
                         plot_fn=lambda *args: None, plot_every=np.inf)
state_indices = np.argmin(scipy.spatial.distance.cdist(path, states), axis=1)
plot_data_and_som(path, states, t=t_max, t_max=t_max, export_name=None)

counter = Counter((s, a) for s, a in zip(state_indices, ap_indices))
p = np.zeros((kS, kA))

for (s, a), val in counter.items():
    p[s, a] = val / len(ap_indices)

# Bayesian imitation
T = 1000
x = np.zeros((T, 3))
x[0] = path[0]

eta = 0.9

for t in range(1, T):
    st = np.argmin(np.linalg.norm(x[t - 1] - states, axis=1), axis=0)
    # we don't need to normalize the probs here, because relatively they stay the same
    at = random.choices(range(kA), weights=p[st,:])
    v1 = action_primitives[at]
    v2 = states[st] - x[t-1]
    x[t] = x[t-1] + eta * v1 + (1-eta) *v2


fig = plt.figure(figsize=(5, 5))
ax = Axes3D(fig)
ax.view_init(elev=45, azim=45)
ax.scatter(path[:, 0], path[:, 1], path[:, 2], alpha=0.1, c='black', marker='o')
ax.plot(x[:, 0], x[:, 1], x[:, 2], c='red', alpha=0.4)
plt.show()