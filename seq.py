import numpy as np
import pandas as pd
from hmmlearn import hmm
from utils import *
import matplotlib.pyplot as plt
import pickle

# configuration
block_size = 30
num_states = 6
transmat_init = np.array([[1/num_states]*num_states]*num_states)
startprob_init = np.array([1, 0, 0, 0, 0, 0])

print('Transition matrix: ')
print(transmat_init)
print('start prob: ')
print(startprob_init)


#set numpy options
#prevent truncated output
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': '{:g}'.format})

#import data from csv
colnames=['date', 'open', 'high', 'low', 'close', 'Adj Close', 'Volume']
X = pd.read_csv('NSEI.csv', names=colnames, header=None)
X = X.iloc[1:, 4].dropna()
#X = X.drop(X.columns[[0]], axis=1)
X = X.values.reshape(-1,1)
print(' ')
print(X.shape)

X_train, X_test = X[:1346,0].reshape(-1,1), X[1347:,0].reshape(-1,1)
#print(X_train)


print(len(X_train))

# init mean and std #
X_init = X_train[:block_size-1,0].reshape(-1, 1).astype('float64')
#print(X_init)
mean = np.mean(X_init)

Nm = np.random.normal(size=(1,num_states))
means_pre = (Nm + mean)

means_init = []
for i in range(num_states):
	means_init.append([means_pre[0][i]])
	
means_init = np.array(means_init)	
	
std_init = np.tile(np.identity(1), (3,1,1))*np.std(X_init)

print('Means: ')
print(means_init)
print('Covars: ')
print(std_init)

####################


train_data = {'x': [], 'aic': [], 'bic': [], 'hqc': [], 'caic': []}

#model = hmm.GaussianHMM(n_components=num_states, transmat_prior=transmat_init, startprob_prior=startprob_init, 
#														covariance_type="full", n_iter=100, init_params='mc')
														
														
model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100, init_params='')

# init the model
model.startprob_ = startprob_init
model.transmat_ = transmat_init
model.means_ = means_init
model.covars_ = std_init
						
#model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100)

#model = init_model.fit(X_train.reshape(-1, 1))
#print(model.score(X))
#exit()

seq = None
lengths = []
# iterate through dataframe
for i in range(0, len(X_train)-block_size + 1):
	print('Iteration {}'.format(i+1))
	df = X_train[i:i+block_size,0].reshape(-1, 1).astype('float64')	
#	print(df)
	if (seq is None):	
		seq = df
		lengths.append(len(df))
	else:
		seq = np.concatenate([seq, df])
		lengths.append(len(df))
		
#model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100, init_params='')	
#	# init the model
#	model.startprob_ = startprob_init
#	model.transmat_ = transmat_init
#	model.means_ = means_init
#	model.covars_ = std_init	
#	
#	# train the model
#model = model.fit(seq, lengths)
#	L = model.score(seq, lengths)
#	train_data['aic'].append(aic(L, 6))
#	train_data['bic'].append(bic(L, 6, block_size))
#	train_data['hqc'].append(hqc(L, 6, block_size))
#	train_data['caic'].append(caic(L, 6, block_size))
#	train_data['x'].append(i)
#	
##	if (i==30):
##		break

#np.save('train_data.npy', train_data)

#plt.plot(train_data['x'], train_data['aic'], color='black')
#plt.xlabel('# iterations')
#plt.ylabel('AIC')
#fig = plt.gcf()
##plt.show()	

#loc = 'train.png'
#fig.savefig(loc, dpi=1000, bbox_inches='tight')

#loc = 'train.eps'
#fig.savefig(loc, format='eps', dpi=1000, bbox_inches='tight')
#					
## Clear the plot
#plt.clf()
#plt.cla()
#plt.close()

#print(seq)
print(len(lengths))
# train the model
model = model.fit(seq, lengths)
print('Transition matrix: ')
print(model.transmat_)

print(model.monitor_.converged)
### save the trained model ###
fname = 'seq.sav'
print('Saving trained model to \'{}\''.format(fname))
pickle.dump(model, open(fname, 'wb'))

#predict
Z2 = model.predict(X)
print(Z2)

#print log likelihood of the model
print(model.score(X))
