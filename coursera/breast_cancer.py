"""

Goal: Understand and visualize how different values of k
      affect the accuracy for a KNearestNeighbors classifier
      for the breast cancer dataset.

"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=[cancer.feature_names])

def unnormalized():
	cancerdf = df
	cancerdf['target'] = cancer.target
	
	y = cancerdf['target']
	X = cancerdf.drop(['target'], axis=1)

	return train_test_split(X, y, test_size=0.75, random_state=0)

def normalized():
	cancerdf = (df - df.mean()) / (df.max() - df.min())
	cancerdf['target'] = cancer.target
	
	y = cancerdf['target']
	X = cancerdf.drop(['target'], axis=1)

	return train_test_split(X, y, test_size=0.75, random_state=0)

def train_and_score(n, X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

X_train_u, X_test_u, y_train_u, y_test_u = unnormalized()
X_train_n, X_test_n, y_train_n, y_test_n = normalized()

n_neighbors = range(1, 100) # tests for n_neighbors between 1 and 10
scores_u = map( lambda n: train_and_score(n, X_train_u, X_test_u, y_train_u, y_test_u), n_neighbors )
scores_n = map( lambda n: train_and_score(n, X_train_n, X_test_n, y_train_n, y_test_n), n_neighbors )

import matplotlib.pyplot as plt

plt.plot(n_neighbors, scores_u)
plt.plot(n_neighbors, scores_n)
plt.legend(['unnormalized', 'normalized'], loc='lower left')
plt.show()