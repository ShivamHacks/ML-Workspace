import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

store = pd.HDFStore('store.h5')

# Training Data

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

y_train = df_train['loss']
test_ids = df_test['id']

print('Train Shape: ' + str(df_train.shape))
print('Test Shape: ' + str(df_test.shape))

df_unprocessed = df_train.drop(['id', 'loss'], axis=1).append(df_test.drop(['id'], axis=1), ignore_index=True)
print('Unprocessed Shape: ' + str(df_unprocessed.shape))

categorical = list(df_unprocessed.select_dtypes(include=['object']).columns)
df_processed = pd.get_dummies(df_unprocessed, columns=categorical)
print('Processed Shape: ' + str(df_processed.shape))

X_train = df_processed.head(len(df_train))
X_test = df_processed.tail(len(df_test))

print('After processing train shape: ' + str(X_train.shape))
print('After processing test shape: ' + str(X_test.shape))

store['X_train'] = X_train
store['X_test'] = X_test
store['y_train'] = y_train
store['test_ids'] = test_ids

"""

y_train = df_train['loss']
X_train = df_train.drop(['id', 'loss'], axis=1)

categorical = list(X_train.select_dtypes(include=['object']).columns)
X_train = pd.get_dummies(X_train, columns=categorical)

train_split = zip(['X_train', 'X_test', 'y_train', 'y_test'], 
	train_test_split(X_train, y_train, test_size=0.75, random_state=0))

# Test Data

test_ids = df_test['id'].tolist()
X_test = pd.get_dummies(df_test.drop(['id'], axis=1), columns=categorical)

# Check Dimensions

print(X_train.shape)
print(X_test.shape) # Column dimension should be same as X_train

# Store Everything
store['train_split'] = train_split
store['test_ids'] = test_ids
store['X_test'] = X_test

"""