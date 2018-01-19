import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Obtain Data
store = pd.HDFStore('store.h5')

X_train = store['X_train']
X_test = store['X_test']
y_train = store['y_train']
test_ids = store['test_ids']

print('Obtained data')

trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.75, random_state=0)

print('Training/Test partition made')

xgtrain = xgb.DMatrix(trainX.values, trainY.values)
xgtest = xgb.DMatrix(testX.values, testY.values)

print('Made DMatrix')

# XGBOOOOOOST

# specify parameters via map, definition are same as c++ version
params = {
	# Learning task parameters
	'objective': 'reg:linear',
	'eval_metric': 'mae',
	'seed': 0,
	'booster': 'gblinear', #"gbtree",# default
	# Tree Booster Parameters
	'subsample': 0.8, # collect 80% of the data only to prevent overfitting
	#'colsample_bytree': 0.8,
	'silent': 1,
	'eta': 0.05
}

# specify validations set to watch performance
watchlist = [(xgtest, 'eval'), (xgtrain, 'train')]
num_round = 100
model = xgb.train(params, xgtrain, num_round, watchlist)

predY = model.predict(xgtest)

print('Made Predictions')

print('Score: ' + str(mean_absolute_error(testY, predY)))

# Save Output
result = list(zip(test_ids.tolist(), predY))
df_submission = pd.DataFrame(result, columns=['id', 'loss'])
df_submission.to_csv('submission.csv', index=False)

print('Saved Submission')

"""
Grid Search Cross Validation takes a long time...

param_grid = {
	'objective': ['reg:linear'],
	'booster': ['gblinear'],
	'subsample': [0.8, 1],
	'colsample_bytree': [0.8, 1],
	'learning_rate': [0.05, 0.1, 0.5],
	'seed': [0],
	'silent': [1]
}

estimator = xgb.XGBRegressor()
regressor = GridSearchCV(estimator=estimator, 
	param_grid=param_grid, 
	scoring='neg_mean_absolute_error', 
	cv=3, refit=True, verbose=10)
regressor.fit(trainX.values, trainY.values)

print('Grid Scores: ' + str(regressor.grid_scores_))
print('Best Estimator: ' + str(regressor.best_estimator_))
print('Best Score: ' + str(regressor.best_score_))
print('Best Params: ' + str(regressor.best_params_))

"""