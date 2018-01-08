import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Obtain Data
store = pd.HDFStore('store.h5')

X_train = store['X_train']
X_test = store['X_test']
y_train = store['y_train']
test_ids = store['test_ids']

print('Loaded Data')

# Make Pipeline
model = Pipeline(steps=[
	('scaler', StandardScaler()),
	('lin_regr', LinearRegression())
])

print('Created Model')

# Fit Model
model.fit(X_train, y_train)

print('Trained Data')

# Make Predictions
y_test = model.predict(X_test)

print('Made Predictions')

# Save Output
result = list(zip(test_ids.tolist(), y_test))
df_submission = pd.DataFrame(result, columns=['id', 'loss'])
df_submission.to_csv('submission.csv', index=False)

print('Saved Submission')

"""
df_train = store['df_train']

y_train = df_train['loss']
X_train = df_train.drop(['id', 'loss'], axis=1)

# Make Pipeline
regr = Pipeline(steps=[
	('scaler', StandardScaler()),
	('lin_regr', LinearRegression())
])

# Fit Model
regr.fit(X_train, y_train)

# Make Predictions
df_test = pd.read_csv('test.csv')

categorical = list(df_test.select_dtypes(include=['object']).columns)
X_test = pd.get_dummies(df_test, columns=categorical)
store['df_test'] = X_test

y_test = regr.predict(X_test.drop(['id'], axis=1))

# Save Output
result = list(zip(X_test['id'], y_test))
df_submission = pd.DataFrame(result, columns=['id', 'loss'])

df_submission.to_csv('submission.csv', index=False)

"""