import os
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from som_utils import plot_som


def load_path_csv(path):
    with open(path) as file:
        items = [line.strip().split(',') for line in file]

    return np.array(items, dtype=np.float)


def plot_data_and_som(data, weights, t, t_max, export_name, dist_fn):
    fig = plt.figure(figsize=(5,5))
    ax = Axes3D(fig)
    ax.view_init(elev=45, azim=45)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1, c='black', marker='o')
    plot_som(weights, dist_fn, ax)
    plt.title(f"k={len(weights)}, iteration {t}/{t_max}")
    plt.savefig(f"p11_results/{export_name}/{export_name}_{t}.png")
    plt.close()


def dist_circle(i, j, k):
    if i > j:
        j, i = i, j
    return min(abs(i - j), abs(k + i - j))


def train_som(data, k, t_max, dist_fn, plot_fn, plot_every):
    def eta(t):
        return 1 - t / t_max

    def sigma(t):
        return np.exp(-t / t_max)

    # precompute all distances and store in a matrix
    dist = np.array([[dist_fn(i, j) for j in range(k)] for i in range(k)]) ** 2

    # initialize weights with random samples
    weights = np.array(random.sample(list(data), k))

    plot_fn(data, weights, 0)

    for t in tqdm(range(t_max)):
        x = random.choice(data)
        i = np.argmin(np.linalg.norm(weights - x, axis=1))
        weights += eta(t) * np.exp(-dist[i,...,None] / (2 * sigma(t))) * (x - weights)

        if (t + 1) % plot_every == 0:
            plot_fn(data, weights, t + 1)

    if t_max % plot_every != 0:
        plot_fn(data, weights, t_max)

    return weights


if __name__ == '__main__':
    paths = {'path1': load_path_csv("../data/q3dm1-path1.csv"),
             'path2': load_path_csv("../data/q3dm1-path2.csv")}

    t_max = 20000

    for path_name, path in paths.items():
        for k in [5,10,20,50]:
            print("run",path_name, k)
            export_name = f"plot_{path_name}_k{k}"
            os.makedirs(f"p11_results/{export_name}", exist_ok=True)

            train_som(path, k, t_max, dist_fn=partial(dist_circle, k=k),
                      plot_fn=partial(plot_data_and_som,
                                      dist_fn=partial(dist_circle, k=k),
                                      export_name=export_name,
                                      t_max=t_max),
                      plot_every=1000)
