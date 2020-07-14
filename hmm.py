import numpy as np
import pandas as pd
from hmmlearn import hmm
from utils import *
import matplotlib.pyplot as plt
import pickle

# configuration
block_size = 100
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


train_data = {'x': [], 'y': []}
curr_model = None
init_model_1 = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100, init_params='')
init_model_2 = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100)
df = X_train[0:0+block_size,0].reshape(-1, 1).astype('float64')	
model = init_model_2.fit(df)
curr_model = init_model_1

# init the model with parameters 
curr_model.startprob_ = startprob_init
curr_model.transmat_ = transmat_init
curr_model.means_ = means_init
curr_model.covars_ = std_init
###############################

# init the model with hmm default parameters 
#curr_model.startprob_ = model.startprob_
#curr_model.transmat_ = model.transmat_
#curr_model.means_ = model.means_
#curr_model.covars_ = model.covars_
###############################

#model = init_model.fit(X_train.reshape(-1, 1))
#print(model.score(X))
#exit()
df = X_train[0:0+block_size,0].reshape(-1, 1).astype('float64')
print('# iterations: {}'.format(len(X_train)-block_size))
# iterate through dataframe
for i in range(0, len(X_train)-block_size + 1):
#	transmat = curr_model.transmat_
#	startprob = curr_model.startprob_
#	means = curr_model.means_
#	covars = curr_model.covars_
#	print('--Iteration {}--'.format(i))
#	print(transmat)
#	print(means)
##	print(covars)
#	print(startprob)

	df = X_train[i:i+block_size,0].reshape(-1, 1).astype('float64')	
	
	try:
		res = aic(model.score(df), 6)
		train_data['y'].append(res)
		train_data['x'].append(i)
	except:
		pass	
	
	try:
		print('Iteration: {}'.format(i+1))
		model = curr_model.fit(df)
	except:
		print('Error')
		
#	print(df)
#	print(np.isnan(df).any())	
	
		
	
	
plt.plot(train_data['x'], train_data['y'], color='black')
plt.xlabel('# iterations')
plt.ylabel('AIC')
fig = plt.gcf()
plt.show()	

loc = 'train.png'
fig.savefig(loc, dpi=1000, bbox_inches='tight')

loc = 'train.eps'
fig.savefig(loc, format='eps', dpi=1000, bbox_inches='tight')
					
# Clear the plot
plt.clf()
plt.cla()
plt.close()

### save the trained model ###
fname = 'hmm.sav'
print('Saving trained model to \'{}\''.format(fname))
pickle.dump(curr_model, open(fname, 'wb'))
	
print(curr_model)
print(curr_model.transmat_)
