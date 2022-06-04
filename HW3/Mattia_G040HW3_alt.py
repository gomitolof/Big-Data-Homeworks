# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(strToVector).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File :", filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", (end - start) * 1e3, " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", (end - start) * 1e3, " ms")
     



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    return tuple(map(float, str.split(",")))



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):
    #------------- ROUND 1 ---------------------------

    tim = time.time()
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    elems = coreset.collect()
    print("Time Round 1:", (time.time() - tim) * 1e3, "ms")
    
    # END OF ROUND 1

    #------------- ROUND 2 ---------------------------

    tim = time.time()
    coresetPoints, coresetWeights = map(np.array, zip(*elems))
    centers = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)
    print("Time Round 2:", (time.time() - tim) * 1e3, "ms")

    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = np.array(list(iter))
    centers, weights = kCenterFFT(partition, points)
    return zip(centers, weights)
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    centers = [random.randint(0, len(points) - 1)]
    dist_centers = np.ones_like(points[:, 0])
    dist_centers *= np.inf
    closest = np.zeros(len(points), dtype=np.int64)

    temp_p = np.ndarray(points.shape)
    temp_d = np.ndarray(dist_centers.shape)

    for i in range(k - 1):
        np.subtract(points, points[centers[-1]], out=temp_p)
        np.square(temp_p, out=temp_p)
        np.sum(temp_p, out=temp_d, axis=1)
        np.minimum(dist_centers, temp_d, out=dist_centers)
        closest[dist_centers == temp_d] = i
        centers.append(dist_centers.argmax())

    np.subtract(points, points[centers[-1]], out=temp_p)
    np.square(temp_p, out=temp_p)
    np.sum(temp_p, out=temp_d, axis=1)
    closest[temp_d < dist_centers] = k - 1

    weights = np.zeros(k)

    for i in range(k):
        weights[i] = (closest == i).sum()

    return points[centers], weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    min_d = np.ones_like(centers[:, 0]) * np.inf
    min_c = np.zeros(len(min_d), dtype=np.int64)

    temp_p = np.zeros_like(points)
    temp_d = np.zeros_like(min_d)

    for i, c in enumerate(centers):
        np.subtract(points, c, out=temp_p)
        np.square(temp_p, out=temp_p)
        np.sum(temp_p, out=temp_d, axis=1)
        np.minimum(min_d, temp_d, out=min_d)
        min_c[min_d == temp_d] = i

    weights = np.zeros_like(min_d)

    for i in range(len(centers)):
        weights[i] = (min_c == i).sum()

    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(P, W, k, z, alpha):
    # d is the (squared) distances matrix, of size n × n.
    d = np.ndarray((len(P), len(P)), dtype=P.dtype)

    # Distances are computed row-by-row, to minimize temporary memory
    # usage.
    for i, l in enumerate(d):
        l[:] = np.square(P - P[i]).sum(1)

    rmin = r = np.sqrt(min(filter(None, d[:k+z+1, :k+z+1].flat)) / 4)
    guesses = 1
	
    # These are temporary n × d matrices and n-vectors of boolean values,
    # allocated here to prevent multiple allocations inside the loop.
    q = np.zeros(d.shape, dtype=bool)
    w = np.zeros(d[0].shape, dtype=bool)

    # Z is represented as a vector; if x in Z in the original algorithm
    # then Z[x] = W[x], otherwise Z[x] = 0.
    Z = np.zeros_like(W)

    while True:
        # The equivalent to Z ← P.
        Z[:] = W
        # S is represented as a list because the algorithm itself
        # makes sure that no element will appear twice, and list
        # handling is quite faster than sets.
        S = []

        lr = (3 + 4 * alpha) * r
        lr *= lr

        # Compute the ball of center x and radius (1 + 2α)r for each
        # x (in each row and column).
        np.less_equal(d, np.square((1 + 2 * alpha) * r), out=q)

        for _ in range(k):
            if not Z.any():
                break

            # This single line computes newcenter; for each line
            # of q (each candidate center), computing the Z-ball
            # weight is here equivalent to indexing Z itself.
            # Note that the following could in theory be replaced
            # by (q @ Z).argmax(), but that implementation is ~18
            # times SLOWER than the one below, as numpy has to
            # convert each element of q from bool to float64
            # internally in order to compute the matmul.
            # Note also that the Z[l] for each line l of q do not
            # in general form a rectangular array, so they cannot
            # be stacked to form one.
            newcenter = np.fromiter((Z[l].sum() for l in q),
                                    Z.dtype).argmax()
            S.append(newcenter)
            # In this formulation, removing each y from Z is
            # equivalent to zeroing the relevant positions in Z.
            Z[np.less_equal(d[newcenter], lr, out=w)] = 0

        # W_Z is not kept, as it is utterly inexpensive to compute on
        # the fly here.
        if Z.sum() <= z:
            print("Initial guess =", rmin)
            print("Final guess =", r)
            print("Number of guesses =", guesses)
            return P[S]

        r *= 2
        guesses += 1



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points, centers, z):
    def dist(P):
        out = np.ones_like(P[:, 0]) * np.inf
        temp_p = np.zeros_like(P)
        temp_d = np.zeros_like(out)

        for c in centers:
            np.subtract(P, c, out=temp_p)
            np.square(temp_p, out=temp_p)
            temp_p.sum(axis=1, out=temp_d)
            np.minimum(out, temp_d, out=out)

        out.sort()
        return out[-z - 1:]

    return np.sqrt(
        points.mapPartitions(lambda x: dist(np.array(list(x))))
              .top(z + 1)[-1]
    )




# Just start the main program
if __name__ == "__main__":
    main()
