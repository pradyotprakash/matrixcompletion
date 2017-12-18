import numpy as np
import math, random, itertools, time
from multiprocessing import Pool

random.seed(1)

def f(t):
	u, i = t
	if M[u, i] == 1:
		return X[u, i]

	l1 = []
	Suiv = {}
	for v in N2[i]:
		intersection = N1[u] & N1[v]
		if v != u and len(intersection) >= beta:
			l1.append(v)
			val = 0.0

			for x in intersection:
				for y in intersection:
					val += ((X[u, x] - X[v, x]) - (X[u, y] - X[v, y]))**2
			Suiv[v] = val/(2*len(intersection)*(len(intersection)-1))

	l2 = []
	Siuj = {}
	for j in N1[u]:
		intersection = N2[i] & N2[j]
		if j != i and len(intersection) >= beta:
			l2.append(j)
			val = 0.0

			for x in intersection:
				for y in intersection:
					val += ((X[x, i] - X[x, j]) - (X[y, i] - X[y, j]))**2
			Siuj[j] = val/(2*len(intersection)*(len(intersection)-1))

	val = 0.0
	weight = 0.0
	for v in l1:
		for j in l2:
			if M[v, j] == 1:
				wvj = math.exp(-lam*min(Suiv[v], Siuj[j]))
				val += wvj*(X[u, j] + X[v, i] - X[v, j])
				weight += wvj
	try:
		return val/weight
	except:
		return 0.0

m, n = 100, 10000
omega_size = 50000

X = np.array([[i+j for j in range(n)] for i in range(m)])
rows, cols = X.shape

row_array = range(rows)
col_array = range(cols)

omega = set(zip([random.randint(0,rows-1) for _ in xrange(omega_size)], 
			[random.randint(0,cols-1) for _ in xrange(omega_size)]))

N1 = [set() for _ in row_array]
N2 = [set() for _ in col_array]

M = np.zeros(shape=X.shape)

for x, y in omega:
	N1[x].add(y)
	N2[y].add(x)
	M[x, y] = 1

# betas = range(2, 6)
# lambdas = map(lambda s:float(s)/10, range(1, 11, 1))
betas = [2]
lambdas = [0.5]

for beta in betas:
	for lam in lambdas:
		start = time.time()
		all_vals = itertools.product(*[row_array, col_array])
		pool = Pool(4)
		ret = list(pool.imap(f, all_vals, rows*cols/4))
		del pool

		X_hat = np.reshape(np.matrix(ret), newshape=(rows, cols))
		totTime = time.time() - start
		print beta, lam, totTime, np.linalg.norm(X - X_hat, ord='fro')/np.linalg.norm(X, ord='fro')
