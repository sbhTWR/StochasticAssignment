import numpy as np
import pandas as pd
from hmmlearn import hmm
from utils import *
#prevent truncated output
#np.set_printoptions(threshold=np.inf)

#import data from csv
colnames=['date', 'open', 'high', 'low', 'close']
X = pd.read_csv('data.csv', names=colnames, header=None)
X = X.iloc[1:]
X = X.drop(X.columns[[0]], axis=1)
X = X.astype(np.float64)
#X = X.values
print(' ')
#print(X)

# states for the Markov chain
class States:
	NUM_STATES = 3
	DECREASE = 0
	SAME = 1
	INCREASE = 2


print(States.NUM_STATES)	
transmat = np.zeros((States.NUM_STATES, States.NUM_STATES))

prev = None
curr_state = None
# X is the dataset containing closing index for each day
for index, row in X.iterrows():
	if (prev is None):
		prev = row['close']
		# continue iteration after assigning prev
		continue
		
	d = row['close'] - prev
    
	if (curr_state is None):
		if (d < 0):
			curr_state = States.DECREASE
		elif (d >0):
			curr_state = States.INCREASE
		else:
			curr_state = States.SAME
		
		prev = row['close'] 
		# continue iteration after assigning state	
		continue		
	
	if (np.isclose(d, 0, atol=10)):
		transmat[curr_state][States.SAME] += 1
		curr_state = States.SAME
	elif (d < 0):
		transmat[curr_state][States.DECREASE] += 1
		curr_state = States.DECREASE
	else:
		transmat[curr_state][States.INCREASE] += 1
		curr_state = States.INCREASE
		
			
	# update prev
	prev = row['close']

transmat = transmat/transmat.sum(axis=1)[:,None]
	
print(transmat)	 
	    			
    
