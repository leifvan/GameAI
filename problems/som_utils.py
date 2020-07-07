import numpy as np


def plot_som(weights, dist_fn, ax):
    """
    Plots a SOM with given weights onto axis ``ax`` and infers connectivity from the given
    ``dist_fn``, assuming the distance to the neighbors is 1.
    """
    k = len(weights)
    dist = np.array([[dist_fn(i, j) for j in range(k)] for i in range(k)])

    for i, wi in enumerate(weights):
        js = [j for j in np.argwhere(dist[i] == 1) if j > i]
        for j in js:
            pair = np.concatenate([wi[None], weights[j,:]], axis=0).T
            ax.plot(*pair, c='blue')

    ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='blue', marker='o')
