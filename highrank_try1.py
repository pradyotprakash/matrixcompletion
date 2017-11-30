def calcPartialDist(a, b, n):
    s = 0
    for i in xrange(n):
        if a[i] != 0 and b[i] != 0:
            s += (a[i] - b[i]) ** 2
    return math.sqrt(s)

import numpy as np
import random, math

# generation of matrix
n = 100
N = 5000
k = 10
r = 5

# ?
# C = 1
# omega_size = int(C * r * N * (math.log(n) ** 2))
omega_size = 400000

s0 = int(3 * k * math.log(k))

# ?
# values of following constants?
# u0 = n / r
u0 = math.log(n)

u1 = math.log(n)

# beta = 1.5625 (by reverse calculating from delta0 = 0.5)
beta = 0.0001 # (works but totally wrong since beta should be > 1, but even min val 1 gives eta0 such that sampling is impossible)

eps0 = 1 # 0.5 does not work

v0 = 0.33 # (by reverse calculating from value of s0 (ignored delta0))

# calc from above

delta0 = (n ** (2 - (2 * math.sqrt(beta)))) * math.log(n)
# delta0 = 0.5

# l0 = (2 * k / (v0 * ( (eps0 / math.sqrt(3)) ** r) ) )
# l0 = int(max(l0, (8 * k * (math.log(s0 / delta0) ) / (n * v0 * ( (eps0 / math.sqrt(3)) ** r) ) ) ))
l0 = 2 # (works!)

eta0 = (64 * beta * max(u1 ** 2, u0) / v0) * r * (math.log(n) ** 2)

# t0 = int( 2 * u0 * u0 * math.log(2 * s0 * l0 * n / delta0) )
t0 = 10

Xactual = np.random.rand(n, N)

for j in range(N):
    Xactual[:, j] = Xactual[:, j] / np.linalg.norm(Xactual[:, j])
X = np.zeros(np.shape(Xactual))

# observed entries
omega = zip([random.randint(0, n - 1) for _ in xrange(omega_size)], [random.randint(0, N - 1) for _ in xrange(omega_size)])
for o in omega:
    X[o[0], o[1]] = Xactual[o[0], o[1]]

# get random seeds
seeds = [random.randint(0, N - 1) for _ in xrange(s0)]

# select seeds with more than eta0 obs
seeds = [s for s in seeds if np.count_nonzero(X[:, s]) >= eta0]
s0 = len(seeds)
print "s0", s0

# ?
# while choosing neighbourhood cols, with or without replacement?
# here done with replacement

# get seed neighbourhoods
seed_neighbourhoods = [set() for _ in xrange(s0)]

for i in range(s0):
    s = X[:, seeds[i]]
    for j in range(N):
        if np.count_nonzero(np.multiply(X[:, j], s)) >= t0:
            seed_neighbourhoods[i].add(j)
    seed_neighbourhoods[i] = random.sample(seed_neighbourhoods[i], int(l0 * n))
    seed_neighbourhoods[i] = set([sn for sn in seed_neighbourhoods[i] if calcPartialDist(X[:, sn], s, n) <= (eps0 / math.sqrt(2))])
    print len(seed_neighbourhoods[i])
    if len(seed_neighbourhoods[i]) > n: # should not be needed
        seed_neighbourhoods[i] = random.sample(seed_neighbourhoods[i], n)
    else:
        seed_neighbourhoods[i] = set()

seed_neighbourhoods = [s for s in seed_neighbourhoods if len(s) != 0]

# thinning not done yet


slen = len(seed_neighbourhoods)


# for i in range(slen):
#     mat = # matrix completion?
#     if np.linalg.matrix_rank(mat) > r:
#         seed_neighbourhoods[i] = set()
# seed_neighbourhoods = [s for s in seed_neighbourhoods if len(s) != 0]
# print len(seed_neighbourhoods)
