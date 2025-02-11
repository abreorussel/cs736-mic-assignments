import numpy as np

def rrmse(A,B):
	rrmse = np.sqrt(np.sum(np.square(A-B))/np.sum(A**2))
	return rrmse


