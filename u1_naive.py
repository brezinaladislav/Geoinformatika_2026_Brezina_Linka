# During the development of this script, AI tool Google Gemini was used.
# Specifically for consulting the algorithm and continuous debugging.

# NAIVE NEAREST NEIGHBOUR SEARCH

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


def getNN(xq, yq, zq, X, Y, Z):
    #Find nearest point and its distance
    dmin = inf
    xn, yn, zn = X[0], Y[0], Z[0]
    
    for i in range(len(X)):
        #Compute distance
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]
        d = (dx*dx + dy*dy + dz * dz)**0.5
        
        #Actualize minimum: distance + coordinates
        if d < dmin and d > 0:
            dmin = d
            xn, yn, zn = X[i], Y[i], Z[i]
    return xn, yn, zn, dmin


def drawPoints(X, Y, Z, bx, transp = 0.2):
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal')

    #Compute sphere scale: 1 pix = 25.4 mm
    scale = 1
    if bx > 0:
        scale = int(bx * bx * 40 * 40)
        
    #Plot points
    ax.scatter(X, Y, Z, s=scale, alpha = transp)

    show()
    

#Load points
X, Y, Z = loadPoints('tree_18.txt')

#Draw points
drawPoints(X, Y, Z, 0, 0.2)

#Start timer
start = perf_counter()

#Create list of distances to nearest neighbor
distance = []

#Call getNN for each point
for i in range(len(X)):

    xn, yn, zn, d = getNN(X[i], Y[i], Z[i], X, Y, Z)

    distance.append(d)

#Compute average distance and density
distance_sum = sum(distance)

daver = distance_sum / len(distance)

rho = 1 / daver**3

#End timer
end = perf_counter()

time = end - start

print("Number of points: ", len(X))
print("Average distance: ", round(daver, 2))
print("Density: ", round(rho, 2))
print("Process time: ", round(time, 2), "s")


#Process time as a function of number of points

size = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]

#Start timer
size_time = []

#Go through all sizes
for test in size:

    x_test = X[:test]
    y_test = Y[:test]
    z_test = Z[:test]

    test_start = perf_counter()

    distance_test = []
    
    # Go through all points
    for i in range(test):
        xn, yn, zn,d = getNN(x_test[i], y_test[i], z_test[i], x_test, y_test, z_test)

        distance_test.append(d)

    #Compute average distance and density
    d_aver = sum(distance_test) / len(distance_test)

    rho= 1 / (d_aver**3)

    #End timer
    test_end = perf_counter()

    time = test_end - test_start
    size_time.append(time)

#Create figure
figure()

plot(size, size_time, marker = 'o')

xlabel("Number of points")
ylabel("Process time [s]")
title("Process time of naive search")
legend()
grid()

show()


