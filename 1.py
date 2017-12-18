def calcPartialDist(a, b, n):
    s = 0
    q = 0
    for i in xrange(n):
        # observed in both
        if a[i] != 0 and b[i] != 0:
            s += (a[i] - b[i]) ** 2
            q += 1
    return (math.sqrt(s) * n / q)

def generateMatrix(n, N, k, r, bases):
    Xactual = np.random.rand(n, N)
    # for each subspace
    for i in range(k):
        # choose r vectors from N(0, In)
        basis = [np.random.normal(0, 1, n) for _ in range(r)]
        basis = np.transpose(basis)

        # project random points onto subspace to get points on subspace
        basist = np.transpose(basis)
        # Pb
        projn = np.dot(np.dot(basis, np.linalg.inv(np.dot(basis.T, basis))), basis.T)
        # generate random points in the subspace and store in X
        for j in range(N / k):
            point = np.random.rand(n)
            ptprojn = np.dot(projn, point)
            Xactual[:, (i * (N / k)) + j] = np.copy(ptprojn)
        bases[i] = basis

    return Xactual

def calcCoherence(X, n):
    r = np.linalg.matrix_rank(X)
    maxval = 1.0
    U, S, V = np.linalg.svd(X, full_matrices=True)
    projn = np.dot(np.dot(U, np.linalg.inv(np.dot(U.T, U))), U.T)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        maxval = max(maxval, np.linalg.norm(np.dot(projn, e)))

    return (float(n) / r) * maxval

def subspacesRefine(subspaces, k, n):
    ln = len(subspaces)
    if ln < k:
        return subspaces

    # sort based on rank
    subspaces.sort(key=lambda s:np.linalg.matrix_rank(s))

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

def thinning(neighbour_seed, seed, n, p0):
    t = 0;
    # t = No of observed entries of the seed column
    for j in xrange(len(seed)):
        if(seed[j] != 0):
            t += 1
    q = t0; # Min t0 observations
    # For every column, draw an independent sample of Y
    for i in xrange(len(neighbour_seed)):
    	# Bernoulli Dist
    	pj =0; z = 1;
    	j =q;
    	while j<= t:
    		fact = math.factorial(t)/(math.factorial(j)*math.factorial(t-j))
    		pj += fact*(math.pow(p0,j))*(math.pow(1-p0,t-j));
    		j+=1;

    	if pj < 0.5: #Y=0
    		j=0
    		# Draw a realization of Z
    		from random import *
    		pzrand = uniform(0,1);
            # z = pzrand
    		while z < t:
    			f = math.factorial(t)/(math.factorial(z)*math.factorial(t-z))
    			pz = f*(math.pow(p0,z))*(math.pow(1-p0,t-z)) / 1-pj;
    			if(pz > pzrand ): #
    				break;
    		# Take z subset of the column and discard rest
    		ct = 0
    		k=0
    		Subset = np.zeros(np.shape(neighbour_seed[i]));
    		length = len(neighbour_seed);
    		while ct < z and k < length:
    			if(neighbour_seed[k]!=0):
    				Subset[k] = neighbour_seed[k]
    				ct+=1
    				k+=1
    		neighbour_seed[i] = np.copy(Subset)

    #print neighbour_seed
    return neighbour_seed;


import numpy as np
import random, math
import matplotlib.pyplot as plt

# generation of matrix
n = 100
N = 5000
k = 10
r = 5

# C = 1
# omega_size = int(C * r * N * (math.log(n) ** 2))
omega_size = 400000

p0 = omega_size/float(n*N)
print "p0", p0
s0 = int(math.ceil(3 * k * math.log(k)))

# coherence values approx 1 or 2 for the matrix, checked by printing
u0 = 2.0
u1 = 1.0

# beta = 1.5625 #(by reverse calculating from delta0 = 0.5)
beta = 2.5 # for low delta

eps0 = 1 # 0.5 gives slen = 0

# v0 = 0.33 # (by reverse calculating from value of s0 (ignored delta0) but doesn't work, slen = 0 ?)
v0 = 1.0

delta0 = (float(n) ** (2 - (2 * math.sqrt(beta)))) * math.log(n)
print "delta0", delta0

l1_ = (2 * float(k) / (v0 * ( (eps0 / math.sqrt(3)) ** r) ) )
l2_ = (8 * float(k) * (math.log(s0 / delta0) ) / (n * v0 * ( (eps0 / math.sqrt(3)) ** r) ) )
l0 = math.ceil(max(l1_, l2_))
# too large, leads to 0 slen
print "l0", l0
l0 = 2.0 # (works!)

eta0 = (64 * beta * max(u1 ** 2, u0) / v0) * r * (math.log(n) ** 2)
# too large, leads to 0 seeds
print "eta0", eta0
eta0 = float(n) / 2 # on avg, 55 elements in a column..

t0 = math.ceil( 2 * u0 * u0 * math.log(2 * s0 * l0 * n / delta0) )
print "t0", t0 # too large, larger than n
t0 = min(t0, eta0 / 2)

# original basis matrices, only to be used later for confirmation
bases = [[] for _ in xrange(k)]

Xactual = generateMatrix(n, N, k, r, bases)
# normalize
for j in range(N):
    Xactual[:, j] = Xactual[:, j] / np.linalg.norm(Xactual[:, j])

X = np.zeros(np.shape(Xactual))

# observed entries
omega = zip([random.randint(0, n - 1) for _ in xrange(omega_size)], [random.randint(0, N - 1) for _ in xrange(omega_size)])

for o in omega:
    X[o[0], o[1]] = Xactual[o[0], o[1]]
# X[omega] = Xactual[omega]

print "rank of Xactual", np.linalg.matrix_rank(Xactual)
print "Actual omega size", len(set(omega)), "out of", n * N  # not all indices generated randomly are unique
print "Coherence", calcCoherence(Xactual, n) # not required, only for confirmation
print "rank of X", np.linalg.matrix_rank(X)

temp = sum(map(np.count_nonzero, X))
print "avg nz in cols", float(temp) / N

# get random seeds
seeds = random.sample(range(N), s0)

# select seeds with more than eta0 obs
seeds = [s for s in seeds if np.count_nonzero(X[:, s]) >= eta0]
s0 = len(seeds)
print "s0", s0

# ?
# while choosing neighbourhood cols, with or without replacement?
# here done with replacement

# get seed neighbourhoods
seed_neighbourhoods = [set() for _ in xrange(s0)]
subspaces = [[] for _ in xrange(s0)]

from fancyimpute import SoftImpute

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

    #Thinning for each neighbourhood associated with a seed
    # seed_neighbourhoods contain the neighbour column indexes to be used for a specific seed column
    l_sn = len(seed_neighbourhoods[i])

    if (l_sn != 0):
        # The entire neighbourhood matrix (for ith seed)is mat:

        # mat gives individual columns of neighbourhood matrix at every run
        mat = np.zeros((n,l_sn));
        for k in range(l_sn):
        	mat[:,k] = np.copy(X[:,seed_neighbourhoods[i][k]])

        #Thinning for each neighbourhood associated with a seed
        mat = thinning(mat, s, n, p0) # returning what ? indices or entire sub matrix

        # perform subspace completion to rank r
        mat[mat == 0] = np.nan
        obj = SoftImpute(max_rank = r, verbose=False)
        mat2 = obj.complete(mat)
        subspaces[i] = mat2

seed_neighbourhoods = [s for s in seed_neighbourhoods if len(s) != 0]
subspaces = [s for s in subspaces if len(s) != 0]

print "no of subspaces", len(subspaces)

# subspace refinement
subspaces = subspacesRefine(subspaces, k, n)

print "no of subspaces", len(subspaces)

# choose only top k subspaces
subspaces = subspaces[:k]

# uncomment the line below to complete the matrix using original basis matrices of the k subspaces
# subspaces = bases[:]

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
        # P = np.dot(np.dot(U, np.linalg.inv(np.dot(U.T, U) )), U.T)

        P = np.dot(np.dot(s, np.linalg.inv(np.dot(s.T, s))), s.T)

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
