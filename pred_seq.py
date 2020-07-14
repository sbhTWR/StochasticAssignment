import numpy as np
import pandas as pd
from hmmlearn import hmm
from utils import *
import matplotlib.pyplot as plt
import pickle

# configuration
block_size = 30
num_states = 3
transmat_init = np.array([[1/num_states]*num_states]*num_states)
startprob_init = np.array([1, 0, 0])

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

print('Past dataset length:')
print(len(X_train))
print('Current dataset length: ')
print(len(X_test))

# construct a sequence of past dataset
seq_l = []
seq = None
lengths = []
# iterate through dataframe
for i in range(0, len(X_train)-block_size + 1):
	print(' Sequence construct iteration {}'.format(i+1))
	df = X_train[i:i+block_size,0].reshape(-1, 1).astype('float64')	
	seq_l.append(df)
#	print(df)
	if (seq is None):	
		seq = df
		lengths.append(len(df))
	else:
		seq = np.concatenate([seq, df])
		lengths.append(len(df))

obs = {'y_pred': [], 'y': [], 'x': []}		
# iterate through the test dataset to predict
for i in range(0, len(X_test)-block_size):
	print(' Predict iteration {}'.format(i+1))
	df = X_test[i:i+block_size,0].reshape(-1, 1).astype('float64')
	
	curr_index = df[-1,0]
	next_index = float(X_test[i+block_size,0])
	# init 
	#print(X_init)
	mean = np.mean(df)

	Nm = np.random.normal(size=(1,num_states))
	means_pre = (Nm + mean)

	means_init = []
	for j in range(num_states):
		means_init.append([means_pre[0][j]])
		
	means_init = np.array(means_init)	
		
	std_init = np.tile(np.identity(1), (3,1,1))*np.std(df)

	print('Means: ')
	print(means_init)
	print('Covars: ')
	print(std_init)
	
	# init the model
	model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100, init_params='')

	# init the model
	model.startprob_ = startprob_init
	model.transmat_ = transmat_init
	model.means_ = means_init
	model.covars_ = std_init
	
	# train the model
	model = model.fit(df)
	
	# calulate curr probability
	curr_prob = model.score(df)
	
	min = np.inf
	min_val = None
	min_seq = None
	next_min_seq = None
	# traverse the sequences 
	for k, s in reversed(list(enumerate(seq_l))):
		past_prob = model.score(s)
#		print('Current: {} Past: {}'.format(curr_prob, past_prob))
		
		diff = abs(curr_prob - past_prob)
		if (k+1 == len(seq_l)):
			continue
			
		if (diff < min):
			min = diff
			min_val = past_prob
			min_seq = s
			next_min_seq = seq_l[k+1]	
#		if (np.isclose(curr_prob, past_prob, atol=10)):
#			found = True
#			print('Found!!')
#			break
	
#	if (found is False):
#		print('Sorry not found...')
#		break
	last_o = min_seq[-1,0]
	last_o_n = next_min_seq[0,0]
	o_diff = last_o_n - last_o
	pred_index = curr_index + o_diff
	print('Diff: {}'.format(o_diff)) 
	print('i: {} Real: {} Prediction {}'.format(i+1, next_index, pred_index))
	obs['y_pred'].append(pred_index)
	obs['y'].append(next_index)
	obs['x'].append(i+1)

# save the observations
np.save('obs.npy', obs)

plt.plot(obs['x'], obs['y_pred'])
plt.plot(obs['x'], obs['y'])
plt.xlabel('Day')
plt.ylabel('Index')
fig = plt.gcf()
#plt.show()	

loc = 'obs.png'
fig.savefig(loc, dpi=1000, bbox_inches='tight')

loc = 'obs.eps'
fig.savefig(loc, format='eps', dpi=1000, bbox_inches='tight')
					
# Clear the plot
plt.clf()
plt.cla()
plt.close()	
			
	
