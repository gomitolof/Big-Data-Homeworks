import sys, time
import numpy as np
import numpy.ma as ma



def SeqWeightedOutliers(P, W, k, z, alpha):
	d = np.ndarray((len(P), len(P)), dtype=P.dtype)
	
	for i, l in enumerate(d):
		l[:] = np.square(P - P[i]).sum(1)

	rmin = r = np.sqrt(min(filter(None, d[:k+z+1, :k+z+1].flat)) / 4)
	guesses = 1
	
	q = np.zeros(d.shape, dtype=bool)
	w = np.zeros(d[0].shape, dtype=bool)
	Z = np.zeros_like(W)

	while True:
		Z[:] = W
		S = []

		lr = (3 + 4 * alpha) * r
		lr *= lr

		np.less_equal(d, np.square((1 + 2 * alpha) * r), out=q)

		for _ in range(k):
			if not Z.any():
				break

			newcenter = np.fromiter((Z[l].sum() for l in q), Z.dtype).argmax()
			S.append(newcenter)
			Z[np.less_equal(d[newcenter], lr, out=w)] = 0

		if Z.sum() <= z:
			return (rmin, r, guesses, S)

		r *= 2
		guesses += 1



def ComputeObjective(P, S, z):
	S = S[3]
	return sorted((np.linalg.norm(P[S] - x, axis=1).min() for x in P), reverse=True)[z]



def main(inputPoints, k, z):
	weights = np.ones(len(inputPoints), dtype=np.float64)
	pre = time.process_time()
	solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
	post = time.process_time()
	objective = ComputeObjective(inputPoints, solution, z)

	print(f"Input size n =  {len(inputPoints)}")
	print(f"Number of centers k =  {k}")
	print(f"Number of outliers z =  {z}")
	print(f"Initial guess =  {solution[0]}")
	print(f"Final guess =  {solution[1]}")
	print(f"Number of guesses =  {solution[2]}")
	print(f"Objective function =  {objective}")
	print(f"Time of SeqWeightedOutliers =  {(post - pre) * 1000}")



if __name__ == "__main__":
	def parse_int(param, name):
		try:
			param = int(param)
		except ValueError:
			print(f"Command line parameter {name} is not an integer: '{param}'", file=sys.stderr)
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
			print(f"Invalid number on line {no+1}", file=sys.stderr)
			exit(-1)

	try:
		inputPoints = np.genfromtxt(fil, delimiter=',')
	except OSError:
		print(f"Input file not found: {fil}", file=sys.stderr)
		exit(-1)
	
	if np.isnan(inputPoints).any():
		print(f"Invalid line in file: {np.where(inputPoints)[0][0]}", file=sys.stderr)
		exit(-1)

	main(inputPoints, k, z)