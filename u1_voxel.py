# During the development of this script, AI tool Google Gemini was used.
# Specifically for consulting the algorithm and continuous debugging.

# ACCELERATED SEARCH USING VOXEL GRID

from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 
from collections import defaultdict
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


def drawPoints(X, Y, Z, bx, transp = 0.2):
    # Create figure
    fig = figure(figsize=(6, 6))
    ax = axes(projection = '3d')
    ax.set_aspect('equal')

    ax.set_box_aspect((1, 1, 0.7))

    #Compute sphere scale: 1 pix = 25.4 mm
    scale = 1
    if bx > 0:
        scale = int(bx * bx * 40 * 40)
        
    #Plot points
    ax.scatter(X, Y, Z, s=scale, alpha = transp)

    show()

    
def drawVoxels(x_min, y_min, z_min, dx, dy, dz, V, n_r):
    # Create figure
    fig = figure(figsize=(6, 6))
    ax = axes(projection = '3d')
    ax.set_aspect('equal') 
    ax.set_box_aspect((1, 1, 0.7))

    #Create meshgrid
    xedges = linspace(x_min, x_min + dx, n_r+1)
    yedges = linspace(y_min, y_min + dy, n_r+1)
    zedges = linspace(z_min, z_min + dz, n_r+1)
    
    VX, VY, VZ = meshgrid(xedges, yedges, zedges,  indexing="ij") 
    
    #Draw voxels
    ax.voxels(VX, VY, VZ, V, edgecolor='k')
    
    show()
    

#Load points
X, Y, Z = loadPoints('tree_18.txt')

#Draw points
drawPoints(X, Y, Z, 0, 0.2)

#Define voxel index function (n_r = None for the purpose of testing for other grid sizes later)
def voxel_index(X, Y, Z, n_r = None):

    # Bounding box
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)
    z_min, z_max = min(Z), max(Z)

    # Dimensions of bounding box
    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min

    # Number of voxels
    n = len(X)
    if n_r is None:
        n_r = int(ceil(n ** (1/3)))

    # Size of one cell
    bx = dx / n_r
    by = dy / n_r
    bz = dz / n_r

    # Create hash table H and array J
    H = defaultdict(list)
    J = []

    # Go through all points
    for i in range(n):

        # Normalize coordinates
        x_norm = (X[i] - x_min) / dx
        y_norm = (Y[i] - y_min) / dy
        z_norm = (Z[i] - z_min) / dz

        # Calculate voxel indices
        jx = int(0.99 * n_r * x_norm)
        jy = int(0.99 * n_r * y_norm)
        jz = int(0.99 * n_r * z_norm)

        # Calculate 1D hash index
        j = jx + jy * n_r + jz * n_r**2

        # Store point index in hash table
        H[j].append(i)

        # Store voxel index for each point
        J.append(j)

    print("Number of points:", n)
    print("Number of voxels:", n_r)
    
    return H, J, n_r, x_min, y_min, z_min, dx, dy, dz, bx, by, bz

H, J, n_r, x_min, y_min, z_min, dx, dy, dz, bx, by, bz = voxel_index(X, Y, Z)

#Define funciton for approximate nearest neighbour search
def annSearch(X, Y, Z, H, J):
    distances = []

    for i in range(len(X)):
        # Get points in the same voxel
        candidates = H[J[i]]

        # Find nearest point among candidates
        dmin = inf
        
        # Calculate distance to candidate point        
        for j in candidates:
            if j == i:
                continue

            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            dz = Z[i] - Z[j]

            d = sqrt(dx**2 + dy**2 + dz**2)

            if d < dmin:
                dmin = d

        if dmin < inf:
            distances.append(dmin)


    # Calculate approximate average distance and density, searching only in the same voxel
    if len(distances) > 0:
        d_aver_grid = sum(distances) / len(distances)
        rho_grid = 1 / (d_aver_grid**3)
    else:
        d_aver_grid = 0
        rho_grid = 0

    return distances, d_aver_grid, rho_grid

#Start timer
t_start = perf_counter()

distances, d_aver_grid, rho_grid = annSearch(X, Y, Z, H, J)

#End timer
t_end = perf_counter()

print("Average distance (grid):", round(d_aver_grid, 2))
print("Density (grid):", round(rho_grid, 2))
print("Execution time:", round(t_end - t_start, 2))

#Calculate curvature for each point using k-nearest neighbours
curvature = []
knn = 30

for i in range(len(X)):
    xnorm = (X[i] - x_min) / dx
    ynorm = (Y[i] - y_min) / dy
    znorm = (Z[i] - z_min) / dz

    jx = int(0.99 * n_r * xnorm)
    jy = int(0.99 * n_r * ynorm)
    jz = int(0.99 * n_r * znorm)

    candidates = []

# Search also neighbouring voxels (3x3x3)
    for kx in range(jx - 1, jx + 2):
        for ky in range(jy - 1, jy + 2):
            for kz in range(jz - 1, jz + 2):

                # Check voxel boundaries
                if (kx < 0 or kx >= n_r or
                    ky < 0 or ky >= n_r or
                    kz < 0 or kz >= n_r):
                    continue

                # Calculate 1D index
                j_neighbor = kx + ky * n_r + kz * n_r**2

                if j_neighbor in H:
                    candidates.extend(H[j_neighbor])

    # Calculate distances to candidate points
    distance = []

    for c in candidates:
        if c == i:
            continue

        d = sqrt(
            (X[i] - X[c])**2 +
            (Y[i] - Y[c])**2 +
            (Z[i] - Z[c])**2
        )

        distance.append((d, c))

    # Sort distances
    distance.sort(key=lambda x: x[0])

    # Select 30 nearest neighbours
    knn_candidates = distance[:knn]

    knn_list = []

    for dist, j in knn_candidates:
        knn_list.append([X[j], Y[j], Z[j]])

    # Check number of neighbours
    if len(knn_list) < 3:
        curvature.append(0)
        continue

    # Calculate covariance matrix
    knn_matrix = cov(knn_list, rowvar=False)

    # Calculate eigenvalues
    eigenval = linalg.eigvalsh(knn_matrix)

    # Sort eigenvalues
    lamb = eigenval

    l1 = lamb[0]
    l2 = lamb[1]
    l3 = lamb[2]

    sum_l = l1 + l2 + l3

    # Calculate curvature
    if sum_l == 0:
        kappa = 0
    else:
        kappa = l1 / sum_l

    curvature.append(kappa)


print("Average curvature:", round(sum(curvature) / len(curvature), 2))


#Create figure
fig = figure(figsize=(6, 6))
ax = axes(projection='3d')

# Plot points with colours based on curvature
vis = ax.scatter(
    X, Y, Z,
    c=curvature,
    cmap='magma_r',
    s=0.5
)

ax.set_box_aspect((1, 1, 0.7))

# Add colour scale
colorbar(vis, ax=ax, label='Curvature')

show()


#Calculation time depending on number of points

test = [50, 100, 250, 500, 1000, 2500,
        5000, 7500, 10000, 15000, 25000]

test_time = []

n_r = 30

for sample in test:

    X_sample = X[:sample]
    Y_sample = Y[:sample]
    Z_sample = Z[:sample]

    #Start timer
    t_start_sample = perf_counter()

    # Create spatial index
    H_sample, J_sample, nr, xm, ym, zm, ddx, ddy, ddz, bbx, bby, bbz = voxel_index(
        X_sample, Y_sample, Z_sample, 30
    )

    for i in range(sample):
            # Get points in the same voxel
            candidates = H_sample[J_sample[i]]
    
            # Find nearest point among candidates
            dmin = inf

            for j in candidates:
                if j == i:
                    continue
    
                dx = X_sample[i] - X_sample[j]
                dy = Y_sample[i] - Y_sample[j]
                dz = Z_sample[i] - Z_sample [j]
    
                d = sqrt(dx**2 + dy**2 + dz**2)
    
                if d < dmin:
                    dmin = d

    #End timer
    t_end_sample = perf_counter()

    test_time.append(t_end_sample - t_start_sample)

    print("Points:", sample, "Time:", round(test_time[-1], 2))


# Draw graph
figure()
plot(test, test_time, marker='o', label='Voxel Grid (n_r=30)')
xlabel("Number of points")
ylabel("Calculation time [s]")
title("Calculation time depending on number of points")
grid(True)
legend()
show()


#Calculation time depending on grid size

test_grid = [5, 10, 15, 20, 25, 30]

test_time_grid = []

for grid_size in test_grid:

    #Start timer
    t_start_grid = perf_counter()

    # Create spatial index
    H_grid, J_grid, nr, xm, ym, zm, ddx, ddy, ddz, bbx, bby, bbz = voxel_index(
        X, Y, Z, grid_size
    )


    for i in range(len(X)):
            # Get points in the same voxel
            candidates = H_grid[J_grid[i]]
    
               
            # Find nearest point among candidates
            dmin = inf

            for j in candidates:
                if j == i:
                  continue
    
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                dz = Z[i] - Z[j]
    
                d = sqrt(dx**2 + dy**2 + dz**2)
    
                if d < dmin:
                    dmin = d

    #End timer
    t_end_grid = perf_counter()

    test_time_grid.append(t_end_grid - t_start_grid)

    print("Grid size:", grid_size, "Time:", round(test_time_grid[-1], 2))

    


# Draw graph
figure()
plot(test_grid, test_time_grid, marker='o', label='Voxel Grid')
xlabel("Number of voxels in one direction")
ylabel("Calculation time [s]")
title("Calculation time depending on grid size")
grid(True)
legend()
show()


