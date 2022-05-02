import sys, math, time

# TODO: try using numpy, which is not available in room Ge



def dist(x, y):
	# return math.hypot(*(x[i] - y[i] for i in range(len(x))))
	out = 0

	for i in range(len(x)):
		diff = x[i] - y[i]
		out += diff * diff

	return math.sqrt(out)



def SeqWeightedOutliers(P, W, k, z, alpha):
	d = tuple(tuple(dist(x, y) for y in P) for x in P)
	rmin = r = min(min(*l[:i], *l[i+1:k+z+1]) for i, l in enumerate(d[:k+z+1])) / 2
	guesses = 1

	def ball(j, r):
		return (i for i in Z if d[j][i] <= r)

	while True:
		Z = set(range(len(P)))
		S = set()
		WZ = sum(W)

		while len(S) < k and WZ > 0:
			mx = 0

			for x in range(len(P)):
				ball_weight = sum(W[i] for i in ball(x, (1 + 2 * alpha) * r))\

				if ball_weight > mx:
					mx = ball_weight
					newcenter = x

			S.add(newcenter)

			for y in tuple(ball(newcenter, (3 + 4 * alpha) * r)):
				Z.discard(y)
				WZ -= W[y]

		if WZ <= z:
			return (rmin, r, guesses, S)

		r *= 2
		guesses += 1



def ComputeObjective(P, S, z):
	S = S[3]
	return sorted((min(dist(x, P[y]) for y in S) for x in P), reverse=True)[z]



def main(inputPoints, k, z):
	weights = [1 for _ in inputPoints]
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
		with open(fil) as f:
			inputPoints = [parse_line(*line) for line in enumerate(f)]
	except FileNotFoundError:
		print(f"Input file not found: {fil}", file=sys.stderr)
		exit(-1)

	if len({len(x) for x in inputPoints}) != 1:
		print(f"Empty file or inconsistent number of dimensions in file", file=sys.stderr)
		exit(-1)

	main(inputPoints, k, z)
