import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

store = pd.HDFStore('store.h5')

# Obtain data
X_train = store['X_train']
y_train = store['y_train']

print('Obtained Data')

# Split Data

trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.75, random_state=0)

# Make Model

model_store = pd.HDFStore('models.h5')

models = [('lin_regr', LinearRegression()), ('log_regr', LogisticRegression())]
for model in models:
	print('Training Model: ' + model[0])
	model[1].fit(trainX, trainY)
	#model_store[model[0]] = model[1]
	predY = model.predict(testX)
	score = mean_absolute_error(predY, testY)
	print('Mean Absolute Error: ' + str(score))

print('Done')