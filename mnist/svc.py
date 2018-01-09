import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from sklearn.svm import SVC

df = pd.read_csv('train.csv')
features = df.columns[1:]
X = preprocessing.normalize(df[features])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print 'loaded data'

clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

print 'trained classifier'

y_pred = clf.predict(X_test)
print 'Accuracy: ',  accuracy_score(y_test, y_pred)