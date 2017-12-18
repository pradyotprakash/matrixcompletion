import numpy as np
import random, math
from fancyimpute import SoftImpute

def calcPartialDist(a, b, n):
    s = 0
    q = 0
    for i in xrange(n):
        # observed in both
        if a[i] != 0 and b[i] != 0:
            s += (a[i] - b[i]) ** 2
            q += 1
    return math.sqrt(s)*n/q

def generateMatrix(n, N, k, r, bases):
    Xactual = np.random.rand(N, n)

    for i in range(k):
        basis = np.matrix([np.random.normal(0, 1, n) for _ in range(r)])
        P = np.dot(np.dot(basis.T, np.linalg.inv(np.dot(basis, basis.T))), basis)
        # generate random points in the subspace and store in X
        for j in range(N//k):
            point = np.random.rand(n)
            x = np.dot(P, point)
            Xactual[i*(N//k) + j, :] = x/np.linalg.norm(x)
        bases[i] = basis
    return Xactual

def calcCoherence(X, n):
    r = np.linalg.matrix_rank(X)
    maxval = 1.0
    _, _, Vt = np.linalg.svd(X, full_matrices=True)
    P = np.dot(np.dot(Vt.T, np.linalg.inv(np.dot(Vt, Vt.T))), Vt)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        maxval = max(maxval, np.linalg.norm(np.dot(P, e)))
    return float(n)*maxval/r

def subspacesRefine(subspaces, k, n):
    ln = len(subspaces)
    if ln < k:
        return subspaces

    # sort based on rank
    subspaces.sort(key=lambda s:np.linalg.matrix_rank(s))

    # add first subspace
    ret = []
    ret.append(subspaces[0])
    mat = subspaces[0]

    oldrank = np.linalg.matrix_rank(mat)

    # for each subspace, add and check rank
    for i in range(1, ln):
        mat_old = mat
        mat = np.vstack((mat, subspaces[i]))
        newrank = np.linalg.matrix_rank(mat)

        # if new rank is more than the old rank, then subspace is independent and add it
        # else reset
        if newrank > oldrank:
            ret.append(subspaces[i])
            oldrank = newrank
        else:
            mat = mat_old
    return ret

def thinning(neighbour_seed, seed, p0):
    n = neighbour_seed.shape[1]
    t = 0
    for j in xrange(n):
        if seed[j] != 0: t += 1

    for i in xrange(neighbour_seed.shape[0]):
        q = 0
        for k in xrange(n):
            if neighbour_seed[i, k] != 0 and seed[k] != 0: q += 1

        rho, j = 0, q

        while j <= t:
            fact = math.factorial(t)/(math.factorial(j)*math.factorial(t-j))
            rho += fact*(math.pow(p0, j))*(math.pow(1 - p0, t - j))
            j += 1

        Y = np.random.binomial(1, rho)
        if Y == 0:
            p = []
            for j in range(q):
                fact = math.factorial(t)/(math.factorial(j)*math.factorial(t - j))
                pz = fact*(math.pow(p0, j))*(math.pow(1 - p0, t - j)) / (1 - rho)
                p.append(pz)
            z = np.random.choice(np.arange(q), p=p)

            # Take a z sized subset of the column and discard rest
            ct = 0
            k = 0
            subset = np.zeros(n)
            
            l = []
            for k in range(n):
                if neighbour_seed[i, k] != 0:
                    l.append(k)
            indices = random.sample(l, z)
            subset[indices] = neighbour_seed[i, indices]
            neighbour_seed[i, :] = subset

    return neighbour_seed

# generation of matrix
n = 100
N = 5000
k = 10
r = 5

# C = 1
# omega_size = int(C * r * N * (math.log(n) ** 2))
omega_size = 400000
# observed entries
omega = set(zip([random.randint(0, N-1) for _ in xrange(omega_size)],
        [random.randint(0, n-1) for _ in xrange(omega_size)]))

# original basis matrices, only to be used later for confirmation
bases = [[] for _ in xrange(k)]

Xactual = generateMatrix(n, N, k, r, bases)
X = np.zeros(shape=Xactual.shape)

p0 = 0
for o in omega:
    X[o[0], o[1]] = Xactual[o[0], o[1]]

print 'Original error: ', np.linalg.norm(Xactual - X, ord='fro')/np.linalg.norm(Xactual, ord='fro')

p0 = len(omega)/float(n*N)
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

print "rank of Xactual", np.linalg.matrix_rank(Xactual)
print "Actual omega size", len(omega), "out of", n * N  # not all indices generated randomly are unique
print "Coherence", calcCoherence(Xactual, n) # not required, only for confirmation
print "rank of X", np.linalg.matrix_rank(X)

temp = sum(map(np.count_nonzero, X))
print "avg nz in cols", float(temp) / N

# get random seeds
seeds = random.sample(range(N), s0)

# select seeds with more than eta0 obs
seeds = [seed for seed in seeds if np.count_nonzero(X[seed, :]) >= eta0]
s0 = len(seeds)
print "s0", s0

# ?
# while choosing neighbourhood cols, with or without replacement?
# here done with replacement

# get seed neighbourhoods
seed_neighbourhoods = [set() for _ in xrange(s0)]
subspaces = [[] for _ in xrange(s0)]

for i in range(s0):
    seed = X[seeds[i], :]

    # choose columns with observations at atleast t0 common places
    for j in range(N):
        if np.count_nonzero(np.multiply(X[j, :], seed)) >= t0:
            seed_neighbourhoods[i].add(j)

    # sample l0*n
    seed_neighbourhoods[i] = random.sample(seed_neighbourhoods[i], int(l0 * n))

    # choose those with partial distance < eps0 / sqrt(2)
    seed_neighbourhoods[i] = set([sn for sn in seed_neighbourhoods[i] if calcPartialDist(X[sn, :], seed, n) <= (eps0 / math.sqrt(2))])

    # DIFFERENT ALGO USED
    # if more than n columns in neighbourhood, sample n
    # else leave as it is for those with atleast r columns

    if len(seed_neighbourhoods[i]) > n: # should not be needed
        seed_neighbourhoods[i] = random.sample(seed_neighbourhoods[i], n)
    else:
        if len(seed_neighbourhoods[i]) < r:
            # fix this?
            seed_neighbourhoods[i] = set()
        else:
            seed_neighbourhoods[i] = list(seed_neighbourhoods[i])

    #Thinning for each neighbourhood associated with a seed
    # seed_neighbourhoods contain the neighbour column indexes to be used for a specific seed column
    l_sn = len(seed_neighbourhoods[i])

    if l_sn != 0:
        # The entire neighbourhood matrix (for ith seed)is mat:

        # mat gives individual columns of neighbourhood matrix at every run
        mat = np.zeros(shape=(l_sn, n))
        for k in range(l_sn):
            mat[k, :] = X[seed_neighbourhoods[i][k], :]

        #Thinning for each neighbourhood associated with a seed
        mat = thinning(mat, seed, p0) # returning what ? indices or entire sub matrix

        # perform subspace completion to rank r
        mat[mat == 0] = np.nan
        obj = SoftImpute(max_rank=r, verbose=False)
        subspaces[i] = obj.complete(mat)

seed_neighbourhoods = [sn for sn in seed_neighbourhoods if len(sn) != 0]
subspaces = [s for s in subspaces if len(s) != 0]

print "no of subspaces", len(subspaces)

# subspace refinement
subspaces = subspacesRefine(subspaces, k, n)

print "no of subspaces", len(subspaces)

# choose only top k subspaces
subspaces = subspaces[:k]

# uncomment the line below to complete the matrix using original basis matrices of the k subspaces
# subspaces = bases[:]

num_subspaces = len(subspaces)

# for each column
for j in range(N):
    col = X[j, :]
    tempomega = filter(lambda x:col[x] != 0, range(n))
    col = col[tempomega]
    val = float('Inf')
    optimalchoice = 0

    # for each subspace
    for i in range(num_subspaces):
        # compute incomplete data residual
        s = subspaces[i][:, tempomega]
        P = np.dot(np.dot(s.T, np.linalg.inv(np.dot(s, s.T))), s)

        temp = np.linalg.norm(col - np.dot(P, col))
        if temp < val:
            val = temp
            optimalchoice = i

    # s = subspaces[optimalchoice]
    # P = np.dot(np.dot(s.T, np.linalg.inv(np.dot(s, s.T))), s)
    # X[j, :] = np.dot(P, X[j, :])

    optimal_basis = bases[optimalchoice][:, tempomega]
    alpha = np.dot(np.dot(np.linalg.inv(np.dot(optimal_basis, optimal_basis.T)), optimal_basis), col)
    X[j, :] = np.dot(alpha, bases[optimalchoice])

print np.linalg.norm(Xactual - X, ord='fro')/np.linalg.norm(Xactual, ord='fro')
