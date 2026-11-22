from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 
from collections import defaultdict
from sklearn.neighbors import KDTree
import datetime
from numpy.linalg import *

def loadPoints(file):
    #Load file
    X, Y, Z = [], [], [] ;

    with open(file) as f:
            for line in f:
                x, y, z = line.split('\t')
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))

    return X, Y, Z

# ==========================================
# 1. VLASTNÍ IMPLEMENTACE KD-TREE
# ==========================================

# KD-TREE NODE
class KDNode:
    def __init__(self, point, index, axis, left=None, right=None):
        self.point = point      # [x, y, z]
        self.index = index      # původní index bodu
        self.axis = axis        # 0 (x), 1 (y), nebo 2 (z)
        self.left = left        # levý potomek
        self.right = right      # pravý potomek


def buildKDTree(indices, pts, depth=0):
    if not indices:
        return None

    axis = depth % 3

    # Seřazení indexů podle aktuální osy a volba mediánu
    indices.sort(key=lambda i: pts[i][axis])
    median_idx = len(indices) // 2

    node_idx = indices[median_idx]
    node_pt = pts[node_idx]

    left_node = buildKDTree(indices[:median_idx], pts, depth + 1)
    right_node = buildKDTree(indices[median_idx + 1:], pts, depth + 1)

    return KDNode(node_pt, node_idx, axis, left_node, right_node)


def knnSearch(node, query_pt, query_idx, k, best_neighbors):
    """
    best_neighbors: seznam dvojic (vzdálenost, index_bodu)
    udržovaný seřazený od nejbližšího k nejvzdálenějšímu (max délka k)
    """
    if node is None:
        return

   # Pokud to není dotazovaný bod sám na sebe, spočti euklidovskou vzdálenost
    if node.index != query_idx:
        dx = query_pt[0] - node.point[0]
        dy = query_pt[1] - node.point[1]
        dz = query_pt[2] - node.point[2]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5

        # Vložení do seznamu k-nejbližších sousedů POUZE pokud d > 0
        if len(best_neighbors) < k and d > 0:
            best_neighbors.append((d, node.index))
            best_neighbors.sort(key=lambda item: item[0])
        elif d > 0 and d < best_neighbors[-1][0]:
            best_neighbors[-1] = (d, node.index)
            best_neighbors.sort(key=lambda item: item[0])

    # Určení primární a sekundární větve stromu
    axis = node.axis
    diff = query_pt[axis] - node.point[axis]

    if diff <= 0:
        first_branch = node.left
        second_branch = node.right
    else:
        first_branch = node.right
        second_branch = node.left

    # Prohledej nejprve nadějnější větev
    knnSearch(first_branch, query_pt, query_idx, k, best_neighbors)

    # Ořezávání: prohledej druhou větev, jen pokud dělící rovina leží blíže než k-tý nejhorší dosavadní soused
    if len(best_neighbors) < k or abs(diff) < best_neighbors[-1][0]:
        knnSearch(second_branch, query_pt, query_idx, k, best_neighbors)


# ==========================================
# 2. VÝPOČET KŘIVOSTI (PCA)
# ==========================================

def computeCurvature(neighbors_pts):
    """
    neighbors_pts: numpy array tvaru (30, 3)
    Vypočte kovarianční matici, seřadí vlastní čísla a spočte kappa
    """
    cov_matrix = np.cov(neighbors_pts, rowvar=False)
    # np.linalg.eigvalsh vrací reálná vlastní čísla seřazená vzestupně: l1 <= l2 <= l3
    eigvals = np.linalg.eigvalsh(cov_matrix)
    l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]

    sum_l = l1 + l2 + l3
    if sum_l > 0:
        return l1 / sum_l
    return 0.0


# ==========================================
# 3. BĚH VÝPOČTU: HUSTOTA A KŘIVOST
# ==========================================

file_path = r"C:\Users\brezi\Documents\UK\Mgr studium\01_zimní\Geoinformatika\úkoly\01_úkol\tree_18.txt"
X, Y, Z = loadPoints(file_path)
pts = list(zip(X, Y, Z))
n_points = len(pts)

print(f"Stavba KD-tree pro {n_points} bodů...")
t0 = perf_counter()
indices = list(range(n_points))
tree_root = buildKDTree(indices, pts)
t1 = perf_counter()
print(f"KD-tree postaven za {t1 - t0:.4f} s")

# A) Výpočet hustoty (k=1)
print("\nVýpočet prostorové hustoty (k=1)...")
start_density = perf_counter()
distances_1nn = []

for i in range(n_points):
    neighbors = []
    knnSearch(tree_root, pts[i], query_idx=i, k=1, best_neighbors=neighbors)
    distances_1nn.append(neighbors[0][0])

daver = sum(distances_1nn) / len(distances_1nn)
rho = 1.0 / (daver ** 3)
end_density = perf_counter()

print("Average distance (daver):", daver)
print("Density (rho):", rho)
print("Process time (hustota):", end_density - start_density, "s")

# B) Výpočet křivosti (k=30)
print("\nVýpočet aproximované křivosti (k=30)...")
start_curv = perf_counter()
curvatures = []

for i in range(n_points):
    neighbors_30 = []
    knnSearch(tree_root, pts[i], query_idx=i, k=30, best_neighbors=neighbors_30)
    
    # Vytažení souřadnic 30 sousedů
    neigh_coords = np.array([pts[item[1]] for item in neighbors_30])
    curvatures.append(computeCurvature(neigh_coords))

end_curv = perf_counter()
print("Process time (křivost):", end_curv - start_curv, "s")


# ==========================================
# 4. VIZUALIZACE BAREVNÉ ŠKÁLY KŘIVOSTI
# ==========================================

fig = figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

scatter_plot = ax.scatter(X, Y, Z, c=curvatures, cmap='viridis', s=2, alpha=0.8)
cbar = fig.colorbar(scatter_plot, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Aproximovaná křivost $\kappa$")

title("Aproximovaná křivost bodového mračna (KD-tree, k=30)")
show() 

# ============================================================
# GRAF: VÝPOČETNÍ ČAS JAKO FUNKCE VELIKOSTI VSTUPNÍ MNOŽINY
# ============================================================

print("\n--- Měření závislosti času na velikosti množiny bodů ---")

# Velikosti testovaných množin bodů (poslední je celé mračno)
test_sizes = [100, 500, 1000, 2500, 5000, 10000, 15000, 20000, n_points]
times_total = []

for n in test_sizes:
    # Výběr prvních n bodů (nebo náhodný výběr)
    pts_subset = pts[:n]
    indices_subset = list(range(n))
    
    t_start = perf_counter()
    
    # 1. Stavba KD-tree
    sub_tree = buildKDTree(indices_subset, pts_subset)
    
    # 2. Hledání 1-NN (pro hustotu)
    for i in range(n):
        neigh = []
        knnSearch(sub_tree, pts_subset[i], query_idx=i, k=1, best_neighbors=neigh)
        
    # 3. Hledání 30-NN a křivost (pouze pokud je bodů >= 30)
    if n >= 30:
        for i in range(n):
            neigh_30 = []
            knnSearch(sub_tree, pts_subset[i], query_idx=i, k=30, best_neighbors=neigh_30)
            coords = np.array([pts_subset[item[1]] for item in neigh_30])
            _ = computeCurvature(coords)
            
    t_elapsed = perf_counter() - t_start
    times_total.append(t_elapsed)
    print(f"N = {n:5d} bodů  ->  Čas: {t_elapsed:.4f} s")

# Vykreslení grafu
figure(figsize=(9, 6))
plot(test_sizes, times_total, marker='o', color='#1f77b4', linewidth=1.5, markersize=6)

title("KD-tree: Výpočetní čas jako funkce velikosti množiny bodů", fontsize=12, pad=12)
xlabel("Počet bodů $N$", fontsize=11)
ylabel("Výpočetní čas [s]", fontsize=11)
grid(True, linestyle='--', alpha=0.6)
tight_layout()

show()