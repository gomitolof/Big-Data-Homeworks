import sys, time
import numpy as np



def SeqWeightedOutliers(P, W, k, z, alpha):
	# P is expected to be a n × d matrix, and W an n-vector.

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
			print(f"Initial guess =  {rmin}")
			print(f"Final guess =  {r}")
			print(f"Number of guesses =  {guesses}")
			return S

		r *= 2
		guesses += 1



def ComputeObjective(P, S, z):
	# This function is not time-critical, and accuracy is a plus, so we
	# just use np.linalg.norm to compute distances.
	return sorted((np.linalg.norm(P[S] - x, axis=1).min() for x in P),
		      reverse=True)[z]



def main(inputPoints, k, z):
	weights = np.ones(len(inputPoints), dtype=np.float64)

	print(f"Input size n =  {len(inputPoints)}")
	print(f"Number of centers k =  {k}")
	print(f"Number of outliers z =  {z}")

	pre = time.process_time()
	solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
	post = time.process_time()
	objective = ComputeObjective(inputPoints, solution, z)

	print(f"Objective function =  {objective}")
	print(f"Time of SeqWeightedOutliers =  {(post - pre) * 1000}")



if __name__ == "__main__":
	def parse_int(param, name):
		try:
			param = int(param)
		except ValueError:
			print(f"Command line parameter {name} is not "
			       "an integer: '{param}'", file=sys.stderr)
			exit(-1)

		return param

	if len(sys.argv) != 4:
		print(f"Usage: {sys.argv[0]} file k z", file=sys.stderr)
		exit(-1)

	_, fil, k, z = sys.argv
	k = parse_int(k, 'k')
	z = parse_int(z, 'z')

	def parse_line(no, line):
		try:
			return tuple(map(float, line.split(',')))
		except ValueError:
			print(f"Invalid number on line {no+1}",
			      file=sys.stderr)
			exit(-1)

	try:
		inputPoints = np.genfromtxt(fil, delimiter=',')
	except OSError:
		print(f"Input file not found: {fil}", file=sys.stderr)
		exit(-1)
	
	if np.isnan(inputPoints).any():
		print(f"Invalid line in file: {np.where(inputPoints)[0][0]}",
		      file=sys.stderr)
		exit(-1)

	main(inputPoints, k, z)
