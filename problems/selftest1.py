import random
from collections import Counter

import numpy as np

# ------------------------------------------
# ----------------- TASK 1 -----------------
# ------------------------------------------

p1 = np.array([[0.3, 0.2, 0.5],
               [0.5, 0.3, 0.2],
               [0.2, 0.5, 0.3]])

def sequence_prob_and_logllh(p, seq):
    transition_probs = [[p1[b,a]] for a,b in zip(seq, seq[1:])]
    prob = np.prod(transition_probs)
    llh = np.sum(np.log(transition_probs))
    return prob, llh

pe1, le1 = sequence_prob_and_logllh(p1, [2,1,0])
print(f"P(e1) = {pe1:.2f}")
print(f"L(e1) = {le1:.2f}")

pe2, le2 = sequence_prob_and_logllh(p1, [2,0,1])
print(f"P(e2) = {pe2:.2f}")
print(f"L(e2) = {le2:.2f}")

# ------------------------------------------
# ----------------- TASK 2 -----------------
# ------------------------------------------

def generate_sequence(p, n, s0):
    s = np.zeros(n, dtype=np.int)
    s[0] = s0
    for i in range(1,n):
        s[i] = random.choices(range(len(p)), weights=p[:,s[i-1]])[0]
    return s


labels = {0: 'A', 1:'B', 2: 'C'}
seq = generate_sequence(p1, 200, 0)
print()
print("Sequence of",len(seq),"items:")
print(''.join(labels[i] for i in seq))

print()
print("average log-likelihood for n=1000 sequences")
sequences = [generate_sequence(p1,200, 0) for _ in range(1000)]
llhs = [sequence_prob_and_logllh(p1, s)[1] for s in sequences]
mean_llh = sum(llhs) / 1000
print(" =",mean_llh)

# ------------------------------------------
# ----------------- TASK 3 -----------------
# ------------------------------------------

transitions = [(j, i) for i,j in zip(seq, seq[1:])]
counter = Counter(transitions)
p_mle = np.zeros((3,3))
for (j,i), num in counter.items():
    p_mle[j,i] = num

for i in range(3):
    p_mle[:, i] /= p_mle[:, i].sum()

print()
print("Maximum likelihood estimate for P using the 200-item sequence above:")
print(np.round(p_mle, 2))

# ------------------------------------------
# ----------------- TASK 4 -----------------
# ------------------------------------------

a = np.zeros((4,3))
a[:3, :3] = np.eye(3) - p1
a[3, :] = 1

b = np.zeros(4)
b[3] = 1

pi = np.linalg.lstsq(a, b)[0]

print()
print("stationary distribution:")
print(np.round(pi, 2))
