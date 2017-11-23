import numpy as np
import math, random, itertools

beta = 2
lam = 1
rows, cols = 1000, 1000
omega_size = 10000

row_array = range(rows)
col_array = range(cols)

# random.seed(1)
# X = np.random.rand(rows, cols)
X = np.array([[i+j for j in col_array] for i in row_array])
omega = set(zip([random.randint(0,rows-1) for _ in xrange(omega_size)], 
			[random.randint(0,cols-1) for _ in xrange(omega_size)]))

N1 = [set() for _ in row_array]
N2 = [set() for _ in col_array]

for x, y in omega:
	N1[x].add(y)
	N2[y].add(x)

X_hat = np.zeros(shape=X.shape)
for u, i in itertools.product(*[row_array, col_array]):
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
			Suiv[v] = val/(2*len(intersection)*len(intersection)-1)

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
			Siuj[j] = val/(2*len(intersection)*len(intersection)-1)

	val = 0.0
	weight = 0.0
	for v, j in itertools.product(*[l1, l2]):
		if (v, j) in omega:
			wvj = math.exp(-lam*min(Suiv[v], Siuj[j]))
			val += wvj*(X[u, j] + X[v, i] - X[v, j])
			weight += wvj
	try:
		X_hat[u, i] = val/weight
	except:
		X_hat[u, i] = 0.0

print np.linalg.norm(X - X_hat, ord='fro')/np.linalg.norm(X, ord='fro')
print X
print X_hat
