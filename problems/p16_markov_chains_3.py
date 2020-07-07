import numpy as np

p2 = np.array([[0.25, 0.1, 0.25],
               [0.50, 0.8, 0.50],
               [0.25, 0.1, 0.25]])

print("Stay in B for tau time steps and then move to another room")
for tau in (5, 10, 100):
    ll = tau * np.log(p2[1, 1]) + np.log(1 - p2[1, 1])
    print(f"- tau = {tau:3d}  =>  L = {ll:7.3f}")

print()
print("Stay in B for tau time steps and then move to C")
for tau in (5, 10, 100):
    ll = tau * np.log(p2[1, 1]) + np.log(p2[2, 1])
    print(f"- tau = {tau:3d}  =>  L = {ll:7.3f}")

print()
print("Start in B and don't move to A for tau time steps")
for tau in (5, 10, 100):
    ll = tau * np.log(1 - p2[0, 1]) + np.log(1 - p2[0, 1] - p2[0, 2])
    print(f"- tau = {tau:3d}  =>  L = {ll:7.3f}")
