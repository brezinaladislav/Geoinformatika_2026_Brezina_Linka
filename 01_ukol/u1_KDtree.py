# omlouváme se za lehce opožděné odevzdání, všiml jsem si chyby ve výpočtu tudíž jsem pdf i tento skript přenahrál znovu
# AI usage: Google Gemini for algorithm consultation and debugging

# ACCELERATED SEARCH USING A KD-TREE

from math import *
from time import *
from matplotlib.pyplot import *
import numpy as np
from collections import defaultdict
import datetime
from numpy.linalg import *

# Load points from file
def loadPoints(file):
    X, Y, Z = [], [], []

    with open(file) as f:
        for line in f:
            x, y, z = line.split('\t')
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))

    return X, Y, Z

# KD-Tree node
class KDNode:
    def __init__(self, point=None, index=None, axis=None, left=None, right=None):
        self.point = point
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right

# KD-Tree
class KDTree:
    def __init__(self, X, Y, Z):
        self.data = []

        for i in range(len(X)):
            self.data.append([X[i], Y[i], Z[i]])
            
        all_indices = list(range(len(self.data)))
        self.root = self.build_tree(all_indices, depth=0)

    # Tree construction
    def build_tree(self, idxs, depth):
        if len(idxs) == 0:
            return None

        axis = depth % 3
        idxs.sort(key=lambda i: self.data[i][axis])
        mid = len(idxs) // 2
        med_idx = idxs[mid]

        left_branch = self.build_tree(idxs[:mid], depth + 1)
        right_branch = self.build_tree(idxs[mid + 1:], depth + 1)

        return KDNode(self.data[med_idx], med_idx, axis, left_branch, right_branch)

    # 1-NN search
    def find_NN(self, query_pt, query_idx):
        best = {"dist": float("inf"), "idx": None}

        def search_nn(node):
            if node is None:
                return

            dx = query_pt[0] - node.point[0]
            dy = query_pt[1] - node.point[1]
            dz = query_pt[2] - node.point[2]
            d = (dx * dx + dy * dy + dz * dz) ** 0.5

            if node.index != query_idx and d > 0 and d < best["dist"]:
                best["dist"] = d
                best["idx"] = node.index

            diff = query_pt[node.axis] - node.point[node.axis]
            if diff < 0:
                near, far = node.left, node.right
            else:
                near, far = node.right, node.left

            search_nn(near)

            # Branch pruning
            if abs(diff) < best["dist"]:
                search_nn(far)

        search_nn(self.root)
        return best["dist"]

    # k-NN search
    def find_knn(self, query_pt, query_idx, k=30):
        neighbors = []

        def search_knn(node):
            if node is None:
                return

            dx = query_pt[0] - node.point[0]
            dy = query_pt[1] - node.point[1]
            dz = query_pt[2] - node.point[2]
            d = (dx * dx + dy * dy + dz * dz) ** 0.5

            if node.index != query_idx and d > 0:
                if len(neighbors) < k:
                    neighbors.append((d, node.index))
                    neighbors.sort(key=lambda item: item[0])
                elif d < neighbors[-1][0]:
                    neighbors[-1] = (d, node.index)
                    neighbors.sort(key=lambda item: item[0])

            diff = query_pt[node.axis] - node.point[node.axis]
            if diff < 0:
                near, far = node.left, node.right
            else:
                near, far = node.right, node.left

            search_knn(near)

            if len(neighbors) < k or abs(diff) < neighbors[-1][0]:
                search_knn(far)

        search_knn(self.root)
        return [idx for d, idx in neighbors]

# Load points from pointcloud
X, Y, Z = loadPoints('tree_18.txt')
pts = list(zip(X, Y, Z))
n_points = len(pts)

# PCA curvature estimation
def computeCurvature(neighbors_pts):
    cov_matrix = np.cov(neighbors_pts, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov_matrix)
    l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]

    sum_l = l1 + l2 + l3
    if sum_l > 0:
        return l1 / sum_l
    return 0.0

# Tree construction
print(f"Tree construction for {n_points} points")
t0 = perf_counter()
indices = list(range(n_points))
tree = KDTree(X, Y, Z)
t1 = perf_counter()
print(f"KD-tree process time: {t1 - t0:.4f} s")

# Density and avarage distance calculation
print("\nDensity calculaion")
start_density = perf_counter()
distances_1nn = []

for i in range(n_points):
    dmin = tree.find_NN(pts[i], i)
    distances_1nn.append(dmin)

daver = sum(distances_1nn) / len(distances_1nn)
rho = 1.0 / (daver ** 3)
end_density = perf_counter()

print("Average distance:", daver)
print("Density:", rho)
print("Process time (density):", end_density - start_density, "s")

# Curvature calculation
print("Curvature calculation")
start_curv = perf_counter()
curvatures = []

for i in range(n_points):
    knn_idxs = tree.find_knn(pts[i], i, k=30)
    neigh_coords = np.array([pts[idx] for idx in knn_idxs])
    curvatures.append(computeCurvature(neigh_coords))

end_curv = perf_counter()
print("Process time (curvature):", end_curv - start_curv, "s")

# DRAW 3D CURVATURE PLOT
print("Curvature plot visualization")
fig = figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal')

# Defining scatter plot and color bar
scatter_plot = ax.scatter(X, Y, Z, c=curvatures, cmap='viridis', s=2)
cbar = fig.colorbar(scatter_plot, ax=ax, shrink=0.6)
cbar.set_label("Curvature")

show() 

# CALCULATION TIME
print("Calculation time on number of points")

test_sizes = [100, 500, 1000, 2500, 5000, 10000, 15000, 20000, n_points]
times_total = []

for n in test_sizes:
    sample_X = X[:n]
    sample_Y = Y[:n]
    sample_Z = Z[:n]
    sample_pts = pts[:n]

    time_start = perf_counter()
    kdtree_sample = KDTree(sample_X, sample_Y, sample_Z)

    for i in range(n):
        dmin = kdtree_sample.find_NN(sample_pts[i], i)

    time_end = perf_counter()
    times_total.append(time_end - time_start)
    print("Points:", n, "Time:", round(times_total[-1], 2), "s")

# DRAW GRAPH
figure()
plot(test_sizes, times_total, marker='o', label='KDTree')
xlabel("Number of points")
ylabel("Calculation time [s]")
title("Calculation time depending on number of points")
grid(True)
legend()
show()
