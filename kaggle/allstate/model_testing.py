# Testing Model on Diabetes dataset

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Obtain data
X, y = datasets.load_diabetes(return_X_y=True)
print X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

model = Pipeline(steps=[
	('scaler', StandardScaler()),
	('regressor', AdaBoostRegressor(LinearRegression(), n_estimators=3, random_state=0))
])

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print('Score: ' + str(score))