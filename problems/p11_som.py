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

path1 = load_path_csv("../data/q3dm1-path1.csv")
path2 = load_path_csv("../data/q3dm1-path2.csv")

# ax = Axes3D(plt.gcf())
# ax.scatter(path1[:,0], path1[:,1], path1[:,2], alpha=0.3)
# plt.show()

# ax = Axes3D(plt.gcf())
# ax.scatter(path2[:,0], path2[:,1], path2[:,2], alpha=0.3)
# plt.show()


def plot_data_and_som(data, weights, t, t_max):
    ax = Axes3D(plt.gcf())
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.2)
    ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='C1')
    ax.plot(weights[:, 0], weights[:, 1], weights[:, 2], c='C1')
    ax.plot(weights[(-1, 0), 0], weights[(-1, 0), 1], weights[(-1,0), 2], c='C1', alpha=1)
    plt.title(f"k={len(weights)}, iteration {t}/{t_max}")


k = 5
path = path2
export_name = f"plot_path2_k{k}"
os.makedirs(f"p11_results/{export_name}", exist_ok=True)


def dist(i,j, k):
    return min(abs(i-j), abs(k+i-j))


t_max = 20000

def eta(t):
    return 1 - t / t_max

def sigma(t):
    return np.exp(-t/t_max)


# initialize weights with random samples from path
weights = np.array(list(random.sample(list(path), k)))

plot_data_and_som(path, weights, 0, t_max)
plt.savefig(f"p11_results/{export_name}/{export_name}_0.png")

for t in tqdm(range(t_max)):
    x = random.choice(path)
    d = np.linalg.norm(weights - x, axis=1)
    i = np.argmin(d)
    for j in range(k):
        weights[j] += eta(t) * np.exp(-dist(i,j,k)/(2*sigma(t))) * (x - weights[j])

    if (t+1) % 1000 == 0:
        plot_data_and_som(path, weights, t+1, t_max)
        plt.savefig(f"p11_results/{export_name}/{export_name}_{t+1}.png")


# plot_data_and_som(path1, weights, t_max, t_max)
# plt.show()