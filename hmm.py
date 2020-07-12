import numpy as np
import pandas as pd
from hmmlearn import hmm
from utils import *

#prevent truncated output
np.set_printoptions(threshold=np.inf)

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

#train the hmm model
model = hmm.GaussianHMM(n_components=6, covariance_type="full", n_iter=100)
model = model.fit(X_train)
print(model)
print(model.transmat_)

#predict
#Z2 = model.predict(X)
#print(Z2)

#print log likelihood of the model
print(model.score(X))
