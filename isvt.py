import numpy as np
import random

m, n = 100, 10000
omega_size = 1000000
num_iter = 20

X = np.random.randn(m, n)
print np.linalg.matrix_rank(X)

omega = zip([random.randint(0,m - 1) for _ in xrange(omega_size)], 
			[random.randint(0,n - 1) for _ in xrange(omega_size)])

X_hat = np.zeros(shape=X.shape)
for x, y in omega:
	X_hat[x, y] = X[x, y]

r = 80
for _ in range(num_iter):
	U, S, V = np.linalg.svd(X_hat, full_matrices=False)
	X_hat = np.dot(np.dot(U[:, :r], np.diag(S[:r])), V[:r, :])
	for x, y in omega:
		X_hat[x, y] = X[x, y]
	print np.linalg.norm(X - X_hat, ord='fro')/np.linalg.norm(X, ord='fro')
