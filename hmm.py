import numpy as np
import pandas as pd
from hmmlearn import hmm
from utils import *

#prevent truncated output
np.set_printoptions(threshold=np.inf)

#import data from csv
colnames=['date', 'open', 'high', 'low', 'close']
X = pd.read_csv('data.csv', names=colnames, header=None)
X = X.iloc[1:]
X = X.drop(X.columns[[0]], axis=1)
X = X.values
print(' ')
#print(X)

#train the hmm model
model = hmm.GaussianHMM(n_components=6, covariance_type="full", n_iter=100)
model = model.fit(X)
print(model)
print(model.transmat_)

#predict
#Z2 = model.predict(X)
#print(Z2)

#print log likelihood of the model
print(model.score(X))
