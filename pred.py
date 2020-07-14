import numpy as np
import pandas as pd
from hmmlearn import hmm
from utils import *
import matplotlib.pyplot as plt
import pickle

#import data from csv
colnames=['date', 'open', 'high', 'low', 'close', 'Adj Close', 'Volume']
X = pd.read_csv('NSEI.csv', names=colnames, header=None)
X = X.iloc[1:, 4].dropna()
#X = X.drop(X.columns[[0]], axis=1)
X = X.values.reshape(-1,1)
print(' ')
print(X.shape)

X_train, X_test = X[:1346,0].reshape(-1,1), X[1347:,0].reshape(-1,1)

# load the trained model
fname = 'seq.sav'
model = pickle.load(open(fname, 'rb'))

# predict
Z = model.predict(X_test.reshape(-1, 1).astype('float64'))
print(Z)
