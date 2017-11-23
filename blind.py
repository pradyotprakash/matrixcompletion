import numpy as np
import math, random, itertools

beta = 2
lam = 0.1
rows, cols = 100, 100
omega_size = 1000

row_array = range(rows)
col_array = range(cols)

# X = np.random.rand(rows, cols)
X = np.array([[i+j for j in col_array] for i in row_array])
omega = zip([random.randint(0,rows-1) for _ in xrange(omega_size)], 
			[random.randint(0,cols-1) for _ in xrange(omega_size)])

M = np.zeros(shape=X.shape)
N1 = [set() for _ in row_array]
N2 = [set() for _ in col_array]
N1uv = [[None for _ in row_array] for _ in row_array]
N2ij = [[None for _ in col_array] for _ in col_array]

for v in omega:
	M[v[0], v[1]] = 1
	N1[v[0]].add(v[1])
	N2[v[1]].add(v[0])

Sb1 = [[None for _ in col_array] for _ in row_array]
Sb2 = [[None for _ in row_array] for _ in col_array]

for u in row_array:
	for i in col_array:
		l = set()
		for v in row_array:
			intersection = N1[u] & N1[v]
			N1uv[u][v] = intersection
			if v != u and (v in N2[i]) and len(intersection) >= beta:
				l.add(v)
		Sb1[u][i] = l

		l = set()
		for j in col_array:
			intersection = N2[i] & N2[j]
			N2ij[i][j] = intersection
			if j != i and (j in N1[u]) and len(intersection) >= beta:
				l.add(j)
		Sb2[i][u] = l

X_hat = np.zeros(shape=X.shape)
Suiv = np.zeros(shape=(rows, cols, rows))
Siuj = np.zeros(shape=(cols, rows, cols))
for u in row_array:
	for i in col_array:
		for v in Sb1[u][i]:
			val = 0.0
			temp = N1uv[u][v]
			for x in temp:
				for y in temp:
					val += ((X[u, x] - X[v, x]) - (X[u, y] - X[v, y]))**2;

			Suiv[u, i, v] = val/(2*len(temp)*len(temp)-1)

		for j in Sb2[i][u]:
			val = 0.0
			temp = N2ij[i][j]
			for x in temp:
				for y in temp:
					val += ((X[x, i] - X[x, j]) - (X[y, i] - X[y, j]))**2;

			Siuj[u, i, j] = val/(2*len(temp)*len(temp)-1)

		val = 0.0
		weight = 0.0
		for v, j in itertools.product(*[Sb1[u][i], Sb2[i][u]]):
			if M[v, j] == 1:
				wvj = math.exp(-lam*min(Suiv[u, i, v], Siuj[i, u, j]))
				val += wvj*(X[u, j] + X[v, i] - X[v, j])
				weight += wvj
		try:
			X_hat[u, i] = val/weight
		except:
			X_hat[u, i] = 0.0

print np.linalg.norm(X - X_hat, ord='fro')/np.linalg.norm(X, ord='fro')
print X
print X_hat
