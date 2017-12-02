def calcPartialDist(a, b, n):
    s = 0
    q = 0
    for i in xrange(n):
        # observed in both
        if a[i] != 0 and b[i] != 0:
            s += (a[i] - b[i]) ** 2
            q += 1
    return (math.sqrt(s) * n / q)

def generateMatrix(n, N, k, r):
    Xactual = np.random.rand(n, N)
    # for each subspace
    for i in range(k):
        # choose r vectors from N(0, In)
        basis = [np.random.normal(0, 1, n) for _ in range(r)]
        basis = np.transpose(basis)

        # project random points onto subspace to get points on subspace
        basist = np.transpose(basis)
        # Pb
        projn = np.dot(np.dot(basis, np.linalg.inv(np.dot(basist, basis) )), basist)
        # generate random points in the subspace and store in X
        for j in range(N / k):
            point = np.random.rand(n)
            ptprojn = np.dot(projn, point)
            Xactual[:, (i * (N / k)) + j] = np.copy(ptprojn)

    return Xactual

def iterativeSVT(X, r, omega):
    Xnew = np.zeros(np.shape(X))
    for o in omega:
        Xnew[o[0], o[1]] = X[o[0], o[1]]

    for _ in xrange(100):
        U, S, V = np.linalg.svd(Xnew, full_matrices=True)
        Snew = np.zeros(np.shape(Xnew))
        for k in range(r):
            Snew[k, k] = S[k]
        Xnew = np.dot(U, np.dot(Snew, V))
        for o in omega:
            Xnew[o[0], o[1]] = X[o[0], o[1]]

    return Xnew

def subspacesRefine(subspaces, k, n):
    ln = len(subspaces)
    if ln < k:
        return subspaces

    # sort based on rank
    subspaces.sort(key = lambda s:np.linalg.matrix_rank(s))

    # add first subspace
    final = []
    final.append(subspaces[0])
    mat = subspaces[0].T

    oldrank = np.linalg.matrix_rank(mat)

    # for each subspace, add and check rank
    for i in range(1, ln):
        s = subspaces[i]

        matold = np.copy(mat)
        for col in subspaces[i].T:
            mat = np.copy(np.vstack((mat, col)))


        newrank = np.linalg.matrix_rank(mat)

        # if new rank is more than the old rank, then subspace is independent and add it
        # else reset
        if newrank > oldrank:
            final.append(subspaces[i])
            oldrank = newrank
        else:
            mat = np.copy(matold)

    return final



import numpy as np
import random, math
import matplotlib.pyplot as plt

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

# beta = 1
beta = 1.5; #1.5625 #(by reverse calculating from delta0 = 0.5)

eps0 = 1 # 0.5 gives slen = 0

# v0 = 0.33 # (by reverse calculating from value of s0 (ignored delta0) but doesn't work, slen = 0 ?)
v0 = 1

# calc from above

delta0 = (n ** (2 - (2 * math.sqrt(beta)))) * math.log(n)
print "delta0", delta0

l0 = (2 * k / (v0 * ( (eps0 / math.sqrt(3)) ** r) ) )
l0 = int(max(l0, (8 * k * (math.log(s0 / delta0) ) / (n * v0 * ( (eps0 / math.sqrt(3)) ** r) ) ) ))
# too large, leads to 0 slen
print "l0", l0
l0 = 2 # (works!)

eta0 = (64 * beta * max(u1 ** 2, u0) / v0) * r * (math.log(n) ** 2)
# too large, leads to 0 seeds
print "eta0", eta0
eta0 = 50 # on avg, 55 elements in a column..

t0 = int( 2 * u0 * u0 * math.log(2 * s0 * l0 * n / delta0) )
# too large, leads to 0 slen
print "t0", t0
t0 = 20 # random

Xactual = generateMatrix(n, N, k, r)
print "rank of Xactual", np.linalg.matrix_rank(Xactual)

# normalize
for j in range(N):
    Xactual[:, j] = Xactual[:, j] / np.linalg.norm(Xactual[:, j])

X = np.zeros(np.shape(Xactual))
# observed entries
omega = zip([random.randint(0, n - 1) for _ in xrange(omega_size)], [random.randint(0, N - 1) for _ in xrange(omega_size)])
for o in omega:
    X[o[0], o[1]] = Xactual[o[0], o[1]]

print "rank of X", np.linalg.matrix_rank(X)

temp = 0.0
for i in range(N):
    temp += np.count_nonzero(X[:, i])

print "avg nz in cols", temp / N

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
    s = np.copy(X[:, seeds[i]])

    # choose columns with obsns at atleast t0 common places
    for j in range(N):
        if np.count_nonzero(np.multiply(X[:, j], s)) >= t0:
            seed_neighbourhoods[i].add(j)

    # sample l0n
    seed_neighbourhoods[i] = random.sample(seed_neighbourhoods[i], int(l0 * n))

    # choose those with partial distance < eps0 / sqrt(2)
    seed_neighbourhoods[i] = set([sn for sn in seed_neighbourhoods[i] if calcPartialDist(X[:, sn], s, n) <= (eps0 / math.sqrt(2))])

    # DIFFERENT ALGO USED
    # if more than n columns in neighbourhood, sample n
    # else leave as it is for those with atleast r columns

    if len(seed_neighbourhoods[i]) > n: # should not be needed
        seed_neighbourhoods[i] = random.sample(seed_neighbourhoods[i], n)
    else:
        if len(seed_neighbourhoods[i]) < r:
            seed_neighbourhoods[i] = set()
        else:
            seed_neighbourhoods[i] = list(seed_neighbourhoods[i])


seed_neighbourhoods = [s for s in seed_neighbourhoods if len(s) != 0]

# thinning not done yet


slen = len(seed_neighbourhoods)
print "slen", slen

subspaces = [[] for _ in range(slen)]

for i in range(slen):
    # get subspace matrix
    templen = len(seed_neighbourhoods[i])
    mat = np.zeros((n, templen))
    for j in range(templen):
        mat[:, j] = np.copy(X[:, seed_neighbourhoods[i][j]])

    # perform subspace completion to rank r
    tempomegalists = np.where(mat != 0)
    tempomega = zip(tempomegalists[0], tempomegalists[1])
    mat2 = iterativeSVT(mat, r, tempomega)

    # check if matrix fits to rank r by checking count of incorrect predictions at observed positions
    U, S, V = np.linalg.svd(mat2, full_matrices=True)
    Snew = np.zeros(np.shape(mat2))
    for kk in range(r):
        Snew[kk, kk] = S[kk]
    mat3 = np.dot(U, np.dot(Snew, V))
    ctr = 0
    for o in tempomega:
        if abs(mat3[o[0], o[1]] - mat[o[0], o[1]]) > pow(10, -2):
            ctr += 1;

    # check that count
    if ctr > len(tempomega) / 5: # random
        seed_neighbourhoods[i] = set()
        subspaces[i] = []
    else:
        subspaces[i] = mat2

seed_neighbourhoods = [s for s in seed_neighbourhoods if len(s) != 0]
subspaces = [s for s in subspaces if len(s) != 0]

print "no of subspaces", len(subspaces)

# subspace refinement
subspaces = subspacesRefine(subspaces, k, n)

print "no of subspaces", len(subspaces)

# choose only top k subspaces
subspaces = subspaces[:k]
slen = len(subspaces)

# for each column
for j in range(N):
    col = np.copy(X[:, j])
    tempomega = filter(lambda x:col[x] != 0, range(n))
    col = col[tempomega]
    val = float('Inf')
    optimalchoice = -1

    # for each subspace
    for i in range(slen):
        # compute incomplete data residual
        s = np.copy(subspaces[i])
        s = np.copy(s.T)
        s = s[:, tempomega]
        s = np.copy(s.T)

        # U, S, V = np.linalg.svd(s, full_matrices=True)
        P = np.dot(np.dot(s, np.linalg.inv(np.dot(s.T, s))), s.T)
        # P = np.dot(np.dot(U, np.linalg.inv(np.dot(U.T, U) )), U.T)

        temp = np.linalg.norm(col - np.dot(P, col))
        if temp < val:
            val = temp
            optimalchoice = i

    # use best fitting subspace for completion of column
    s = np.copy(subspaces[optimalchoice])
    P = np.dot(np.dot(s, np.linalg.inv(np.dot(s.T, s))), s.T)
    colnew = np.dot(P, X[:, j])
    X[:, j] = np.copy(colnew)

print np.linalg.norm(Xactual - X, ord='fro')/np.linalg.norm(Xactual, ord='fro')
# print s1 / s2
