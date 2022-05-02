import time
import sys
import math
import os
import numpy as np

def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result
    
# Euclidean distance between two points
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)

# Euclidean distance between two points without the sqrt
def euclideanWithoutSqrt(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res

def SeqWeightedOutliers(P, W, k, z, alpha):
    N = len(P)
    distances = {}
    ind = 0
    min_dist = 1000
    # create the dictionary that contains all the euclidean distances between points
    # without computing the square root
    distances = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            dist = euclideanWithoutSqrt(P[i], P[j])
            distances[i][j] = dist
            distances[j][i] = dist
            if ind <= k + z:
                if dist < min_dist:
                    min_dist = dist
                ind += 1
    # r shoudl be equal to (the min euclidean distance between first k + z + 1 points)/2
    r = math.sqrt(min_dist)/2
    print("Initial guess =", r)
    Wz = z + 1
    guesses = 0
    W_tot = sum(W)
    while Wz > z:
        Z = dict.fromkeys(P, True)
        S = []
        Wz = W_tot
        small_radious = (1 + 2*alpha) * r
        small_radious = small_radious * small_radious
        big_radious = (3 + 4*alpha) * r
        big_radious = big_radious * big_radious
        guesses += 1
        while len(S) < k and Wz > 0:
            max_ball_weight = 0
            new_center_ind = -1
            # search one of the k centers which maximise the weight of the points around it
            for point_ind in range(N):
                ball_weight = 0
                # compute the weight of the small ball with center "point"
                for uncov_point_ind in range(N):
                    uncov_point = P[uncov_point_ind]
                    if Z[uncov_point] == True and distances[point_ind][uncov_point_ind] <= small_radious:
                        ball_weight += W[uncov_point_ind]
                if ball_weight > max_ball_weight:
                    max_ball_weight = ball_weight
                    new_center_ind = point_ind
            if new_center_ind != -1:
                # add the new center found in S, remove the points inside the largest ball from Z and update the total weights
                new_center = P[new_center_ind]
                S.append(new_center)
                for uncov_point_ind in range(N):
                    uncov_point = P[uncov_point_ind]
                    if Z[uncov_point] == True and distances[new_center_ind][uncov_point_ind] <= big_radious:
                        Z[uncov_point] = False
                        Wz -= W[uncov_point_ind]
            else:
                break
        if Wz > z:
            r = 2 * r
    print("Final guess =", r)
    print("Number of guesses =", guesses)
    return S

# Computes the value of the objective function for the set of points P, the set of centers S,
# and z outliers (the number of centers, which is the size of S, is not needed as a parameter).
# Hint: you may compute all distances d(x,S), for every x in P, sort them, exclude the z largest
# distances, and return the largest among the remaining ones. Note that in this case we are not
# using weights!
def ComputeObjective(P,S,z):
    distances = {}
    ind = 0
    for p in P:
        min_dist = 1000
        for s in S:
            dist = euclideanWithoutSqrt(p, s)
            if min_dist > dist:
                min_dist = dist
        distances[ind] = math.sqrt(min_dist)
        ind += 1
    if z > 0:
        distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
        for i in range(z):
            distances.popitem()
    return max(distances.values())


def main():
# CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python3 G040HW2.py <file_path> <k> <z>"

    # INPUT READING

    # 1. Read input file
    data_path = sys.argv[1]
    assert isinstance(data_path, str), "S must be a string"
    data_path = str(data_path)
    assert os.path.isfile(data_path), "File or folder not found"

    # 2. Read number of centers
    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)

    # 3. Read number of allowed outliers
    z = sys.argv[3]
    assert z.isdigit(), "z must be an integer"
    z = int(z)

    # Read the points in the input file into a list of tuple in Python
    # called inputPoints.
    inputPoints = readVectorsSeq(data_path)

    # Create a list of integer in Python called weights of the same
    # cardinality of inputPoints, initialized with all 1's.
    weights = [1] * len(inputPoints)

    print("Input size n =", len(inputPoints))
    print("Number of centers k =", k)
    print("Number of outliers z =", z)

    # Run SeqWeightedOutliers(inputPoints,weights,k,z,0) to compute a
    # set of (at most) k centers. The output of the method must be
    # saved into a list of tuple in Python called solution.
    start_time = time.time()
    solution = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    time_spent = time.time() - start_time

    objective = ComputeObjective(inputPoints,solution,z)
    print("Objective function =", objective)
    print("Time of SeqWeightedOutliers =", time_spent * 1000)

if __name__ == "__main__":
    main()