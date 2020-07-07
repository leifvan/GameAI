import os
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from p11_som import load_path_csv, train_som, dist_circle


# def dist_circle(i, j, k):
#     if i > j:
#         j, i = i, j
#     return min(abs(i - j), abs(k // 2 + i - j))


def plot_data_and_som(data, weights, t, export_name):
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.view_init(elev=45, azim=45)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1, c='black', marker='o')

    ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='blue', marker='o')

    ax.plot(weights[:k2, 0], weights[:k2, 1], weights[:k2, 2], c='blue', marker='o')
    ax.plot(weights[(k2 - 1, 0), 0], weights[(k2 - 1, 0), 1], weights[(k2 - 1, 0), 2], c='blue', marker='o')
    ax.plot(weights[k2:, 0], weights[k2:, 1], weights[k2:, 2], c='blue', marker='o')
    ax.plot(weights[(-1, k2), 0], weights[(-1, k2), 1], weights[(-1, k2), 2], c='blue', marker='o')
    ax.plot(weights[(k2, 0), 0], weights[(k2, 0), 1], weights[(k2, 0), 2], c='blue', marker='o')

    plt.title(f"k={len(weights)}, iteration {t}/{t_max}")
    plt.savefig(f"p17_results/{export_name}/{export_name}_{t}.png")
    plt.close()


def batch_train_som(data, k, t_max, dist_fn, plot_fn, plot_every):
    def sigma(t):
        return np.exp(-t / t_max)

    # precompute all distances and store in a matrix
    dist = np.array([[dist_fn(i, j) for j in range(k)] for i in range(k)]) ** 2

    # initialize weights with random samples
    weights = np.array(random.sample(list(data), k))

    plot_fn(data, weights, 0)

    for t in tqdm(range(t_max)):
        indices = np.argmin(scipy.spatial.distance.cdist(data, weights), axis=1)
        h = np.exp(-dist[indices] / (2 * sigma(t)))
        weights = np.sum(data[:, None] * h[..., None], axis=0) / np.sum(h, axis=0)[..., None]

        if (t + 1) % plot_every == 0:
            plot_fn(data, weights, t + 1)

    if t_max % plot_every != 0:
        plot_fn(data, weights, t_max)


# batch_train_som(path, k, t_max, dist_fn=dist, plot_fn=plot_data_and_som, plot_every=1000)


if __name__ == '__main__':
    path = load_path_csv("../data/q3dm1-path2.csv")

    for k in [6,10,20,50]:
        print("k =",k)
        k2 = k // 2
        t_max = 20000
        export_name = f"plot_k{k}"
        os.makedirs(f"p17_results/{export_name}", exist_ok=True)


        def dist(i, j):
            if i < k2 and j < k2 or i >= k2 and j >= k2:  # both on one ring
                return dist_circle(i, j, k2)
            else:  # on different rings
                ci, cj = i // k2, (j // k2) * k2  # 0 if on left right, k2 if on right ring
                return dist_circle(i, ci, k2) + dist_circle(j, cj, k2) + 1


        # ---------
        # task 17.1
        # ---------

        export_name = f"plot_k{k}"
        os.makedirs(f"p17_results/{export_name}", exist_ok=True)

        train_som(path, k, t_max, dist_fn=dist,
                  plot_fn=partial(plot_data_and_som, export_name=export_name),
                  plot_every=1000)

        # ---------
        # task 17.2
        # ---------

        export_name = f"plot_batch_k{k}"
        os.makedirs(f"p17_results/{export_name}", exist_ok=True)

        batch_train_som(path, k, t_max, dist_fn=dist,
                        plot_fn=partial(plot_data_and_som, export_name=export_name),
                        plot_every=1000)
