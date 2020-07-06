import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from p11_som import train_som, load_path_csv

path = load_path_csv("../data/q3dm1-path2.csv")

k = 20
k2 = k // 2
t_max = 20000
export_name = f"plot_k{k}"

os.makedirs(f"p17_results/{export_name}", exist_ok=True)


def dist_circle(i, j):
    if i > j:
        j, i = i, j
    return min(abs(i - j), abs(k//2 + i - j))


def plot_data_and_som(data, weights, t):
    fig = plt.figure(figsize=(5,5))
    ax = Axes3D(fig)
    ax.view_init(elev=45, azim=45)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1, c='black', marker='o')

    ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='blue', marker='o')

    ax.plot(weights[:k2, 0], weights[:k2, 1], weights[:k2, 2], c='blue', marker='o')
    ax.plot(weights[(k2-1, 0), 0], weights[(k2-1, 0), 1], weights[(k2-1, 0), 2], c='blue', marker='o')
    ax.plot(weights[k2:, 0], weights[k2:, 1], weights[k2:, 2], c='blue', marker='o')
    ax.plot(weights[(-1, k2), 0], weights[(-1, k2), 1], weights[(-1, k2), 2], c='blue', marker='o')
    ax.plot(weights[(k2, 0), 0], weights[(k2, 0), 1], weights[(k2, 0), 2], c='blue', marker='o')

    plt.title(f"k={len(weights)}, iteration {t}/{t_max}")
    # plt.show()
    plt.savefig(f"p17_results/{export_name}/{export_name}_{t}.png")
    plt.close()


def dist(i, j):
    if i < k2 and j < k2 or i >= k2 and j >= k2:  # both on one ring
        return dist_circle(i, j)
    else:  # on different rings
        ci, cj = i // k2, (j // k2) * k2  # 0 if on left right, k2 if on right ring
        return dist_circle(i, ci) + dist_circle(j, cj) + 1

train_som(path, k, t_max, dist_fn=dist, plot_fn=plot_data_and_som, plot_every=1000)