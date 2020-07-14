import numpy as np

# calculates the Akaike information criterion (AIC)
# L: likelihood function for the model
# N: is the number of states

def aic(L, N, hmm=True):
	
	# number of degrees of freedom
	if hmm is True:
		k = np.power(N, float(2)) + 2*N - 1
	else:
		# for estimating a transition matrix
		# we are calculating N*(N-1), a value
		# for each cell in a matrix of shape
		# N*N, with the constraint that each row
		# should add upto 1. So N-1 for each row
		k = N*(N-1)
	
	# calculate AIC		
	aic = -2*L + 2*k
	return aic

# Bayesian information criterion (BIC)
# L: likelihood function for the model
# M: Number of observation points
def bic(L, N, M, hmm=True):
	# number of degrees of freedom
	if hmm is True:
		k = np.power(N, float(2)) + 2*N - 1
	else:
		# for estimating a transition matrix
		# we are calculating N*(N-1), a value
		# for each cell in a matrix of shape
		# N*N, with the constraint that each row
		# should add upto 1. So N-1 for each row
		k = N*(N-1)
		
	bic = -2*L + k*np.log(M)
	return bic	

#Hannan–Quinn information criterion (HQC)	
# L: likelihood function for the model
# M: Number of observation points
def hqc(L, N, M, hmm=True):

	# number of degrees of freedom
	if hmm is True:
		k = np.power(N, float(2)) + 2*N - 1
	else:
		# for estimating a transition matrix
		# we are calculating N*(N-1), a value
		# for each cell in a matrix of shape
		# N*N, with the constraint that each row
		# should add upto 1. So N-1 for each row
		k = N*(N-1)
		
	hqc = -2*L + k*np.log(np.log(M))
	return hqc	

#Bozdogan Consistent Akaike Information Criterion (CAIC)
# L: likelihood function for the model
# N: is the number of states
# M: Number of observation points
def caic(L, N, M, hmm=True):
	# number of degrees of freedom
	if hmm is True:
		k = np.power(N, float(2)) + 2*N - 1
	else:
		# for estimating a transition matrix
		# we are calculating N*(N-1), a value
		# for each cell in a matrix of shape
		# N*N, with the constraint that each row
		# should add upto 1. So N-1 for each row
		k = N*(N-1)
		
	caic = -2*L + k*(np.log(M)+ 1)
	return caic		
