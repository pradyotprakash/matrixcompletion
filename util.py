import numpy as np
import scipy.linalg as linalg

def get_skew_matrix(l):
	m = np.matrix([[0, l[0], l[1]], [-l[0], 0, l[2]], [-l[1], -l[2], 0]])
	return m

def get_rotation_matrix(l):
	m = get_skew_matrix(l)
	return linalg.expm(m)
