import os
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def load_path_csv(path):
    with open(path) as file:
        items = [line.strip().split(',') for line in file]

    return np.array(items, dtype=np.float)


def plot_data_and_som(data, weights, t):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1, c='black', marker='o')
    ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='blue', marker='o')
    ax.plot(weights[:, 0], weights[:, 1], weights[:, 2], c='blue', marker='o')
    ax.plot(weights[(-1, 0), 0], weights[(-1, 0), 1], weights[(-1, 0), 2], c='blue', marker='o')
    plt.title(f"k={len(weights)}, iteration {t}/{t_max}")
    plt.savefig(f"p11_results/{export_name}/{export_name}_{t}.png")
    plt.close()


def train_som(data, k, t_max, dist_fn, plot_fn, plot_every):
    def eta(t):
        return 1 - t / t_max

    def sigma(t):
        return np.exp(-t / t_max)

    # initialize weights with random samples
    weights = np.array(random.sample(list(data), k))

    plot_fn(data, weights, 0)

    for t in tqdm(range(t_max)):
        x = random.choice(data)
        i = np.argmin(np.linalg.norm(weights - x, axis=1))

        for j in range(k):
            weights[j] += eta(t) * np.exp(-dist_fn(i, j) / (2 * sigma(t))) * (x - weights[j])

        if (t + 1) % plot_every == 0:
            plot_fn(data, weights, t + 1)

    if t_max % plot_every != 0:
        plot_fn(data, weights, t_max)


if __name__ == '__main__':
    path1 = load_path_csv("../data/q3dm1-path1.csv")
    path2 = load_path_csv("../data/q3dm1-path2.csv")

    k = 50
    path = path1
    export_name = f"plot_path1_k{k}"
    os.makedirs(f"p11_results/{export_name}", exist_ok=True)


    def dist(i, j):
        if i > j:
            j, i = i, j
        return min(abs(i - j), abs(k + i - j))


    t_max = 20000


    def plot_data_and_som(weights, t):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(path[:, 0], path[:, 1], path[:, 2], alpha=0.1, c='black', marker='o')
        ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='blue', marker='o')
        ax.plot(weights[:, 0], weights[:, 1], weights[:, 2], c='blue', marker='o')
        ax.plot(weights[(-1, 0), 0], weights[(-1, 0), 1], weights[(-1, 0), 2], c='blue', marker='o')
        plt.title(f"k={len(weights)}, iteration {t}/{t_max}")
        plt.savefig(f"p11_results/{export_name}/{export_name}_{t}.png")
        plt.close()


    def train_som(data, k, t_max, plot_fn, plot_every):
        def eta(t):
            return 1 - t / t_max

        def sigma(t):
            return np.exp(-t / t_max)

        # initialize weights with random samples
        weights = np.array(random.sample(list(data), k))

        plot_fn(weights, 0)

        for t in tqdm(range(t_max)):
            x = random.choice(data)
            i = np.argmin(np.linalg.norm(weights - x, axis=1))

            for j in range(k):
                weights[j] += eta(t) * np.exp(-dist(i, j) / (2 * sigma(t))) * (x - weights[j])

            if (t + 1) % plot_every == 0:
                plot_fn(weights, t + 1)

        if t_max % plot_every != 0:
            plot_fn(weights, t_max)


    train_som(path, k, t_max, dist_fn=dist, plot_fn=plot_data_and_som, plot_every=1000)
