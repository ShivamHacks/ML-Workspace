# http://colah.github.io/posts/2014-10-Visualizing-MNIST/

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

df = pd.read_csv('train.csv')[:10000]
features = df.columns[1:]
X = preprocessing.normalize(df[features])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print 'loaded data'

clf = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
clf.fit(X_train, y_train)

print 'trained classifier'

y_pred = clf.predict(X_test)
print 'Accuracy: ',  accuracy_score(y_test, y_pred)