"""

Achieved a score of 0.975 on Kaggle

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

df_train = pd.read_csv('../input/train.csv')
features = df_train.columns[1:]
X_train = preprocessing.normalize(df_train[features])
y_train = df_train['label']

clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
clf.fit(X_train, y_train)

X_test = preprocessing.normalize(pd.read_csv('../input/test.csv'))
y_test = clf.predict(X_test)

result = list(zip(range(1, len(X_test) + 1), y_test))
df_test = pd.DataFrame(result, columns=['ImageID', 'Label'])

df_test.to_csv('submission.csv', index=False)