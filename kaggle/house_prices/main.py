import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model

from sklearn.metrics import accuracy_score


def retrieveAndCleanData(file):
	data = pd.read_csv(file, header=0)

	# need to fix this

	data = data.fillna(0)

	number = LabelEncoder()
	categorical = list(data.select_dtypes(include=['object']).columns)

	for col in categorical:
		data[col] = number.fit_transform(data[col].astype('str'))

	return data


train = retrieveAndCleanData('train.csv')

trainY = train['SalePrice']
trainX = train.drop(['Id','SalePrice'], axis=1)

# Score the dataset

trainX_1, trainX_2, trainY_1, trainY_2 = train_test_split(trainX, trainY, test_size=0.1, random_state=42)

regr = linear_model.LinearRegression()

regr.fit(trainX_1, trainY_1)
print regr.score(trainX_2, trainY_2)

# regr.fit(trainX, trainY)

# Make Predictions

test = retrieveAndCleanData('test.csv')
testX = test.drop(['Id'], axis=1).values

output = np.column_stack((test['Id'].values, regr.predict(testX)))
results = pd.DataFrame(output.astype('int'), columns=['Id','SalePrice'])

results.to_csv('submission.csv', index=False)

# print results