from collections import Counter

import numpy as np
from tabulate import tabulate

print("+++++++++")
print("task 14.1")
print("+++++++++")
print()

print("   p2   ")
print("--------")


def print_matrix_powers(p, n):
    p_t = p.copy()
    for i in range(n):
        print("t =", 2 ** i)
        print(p_t)
        p_t = p_t @ p_t


p2 = np.array([[0.25, 0.1, 0.25],
               [0.50, 0.8, 0.50],
               [0.25, 0.1, 0.25]])

print_matrix_powers(p2, 5)

print()
print("   p1   ")
print("--------")

p1 = np.array([[0.3, 0.2, 0.5],
               [0.5, 0.3, 0.2],
               [0.2, 0.5, 0.3]])

print_matrix_powers(p1, 5)

print()
print("spectral decompositions")
print("-----------------------")
print()

w2, v2 = np.linalg.eig(p2)
print(w2)
print(v2[:, 0] / np.sum(v2[:, 0]))
print()

w1, v1 = np.linalg.eig(p1)
print(w1)
print(v1[:, 0] / np.sum(v1[:, 0]))

# observe: the eigenvector to eigenvalue 1 are the vectors the MP seems to converge to

print()
print("+++++++++")
print("task 14.2")
print("+++++++++")
print()

# START code from pdf
states = ['A', 'B', 'C']
indices = range(len(states))
state2index = dict(zip(states, indices))
index2state = dict(zip(indices, states))


def generateStateSequence(X0, P, tau):
    sseq = [X0]
    iold = state2index[X0]
    for t in range(tau):
        inew = np.random.choice(indices, p=P[:, iold])
        sseq.append(index2state[inew])
        iold = inew
    return sseq


# END code from pdf

last_elements = [generateStateSequence('B', p2, 9)[-1] for _ in range(10000)]
counter = Counter(last_elements)

histogram_data = [(item, abs_freq, abs_freq / 10000) for item, abs_freq in counter.most_common()]
print(tabulate(histogram_data, headers=("item", "abs freq", "rel freq")))
