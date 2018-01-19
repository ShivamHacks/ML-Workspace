import pandas as pd
import numpy as np

# retrieve and clean data function
def cleanData(csvFile):
	data = pd.read_csv(csvFile, header=0)
	data = data.drop(['Name','Ticket','Cabin'], axis=1) # drop useless columns

	dummies = [] # convert categorical variable to indicator variable
	for col in ['Pclass','Sex','Embarked']:
		dummies.append(pd.get_dummies(data[col]))

	titanic_dummies = pd.concat(dummies, axis=1)
	data = pd.concat((data, titanic_dummies), axis=1)
	data = data.drop(['Pclass','Sex','Embarked'], axis=1)
	data['Age'] = data['Age'].interpolate() # fill in missing ages
	return data

# Retrieve and Clean the data

trainData = cleanData('train.csv')
trainY = trainData['Survived'].values
trainData = trainData.drop(['Survived'], axis=1)
trainX = trainData.values

# Train the model and score it


from sklearn import tree
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB

classifierNames = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Naive Bayes']
classifiers = [];
classifiers.append(tree.DecisionTreeClassifier(max_depth=5))
classifiers.append(ensemble.RandomForestClassifier(n_estimators=100))
classifiers.append(ensemble.GradientBoostingClassifier())
classifiers.append(GaussianNB())

scores = []
highestScore = 0;
highestClf = 0;
for index, classifier in enumerate(classifiers):
	classifier.fit(trainX, trainY)
	score = classifier.score(trainX, trainY)
	print 'Classifier: ' + classifierNames[index] + ', Score=' + str(score)
	highestClf = index if score > highestScore else highestClf
	highestScore = score if score > highestScore else highestScore

print 'Best Classifier: ' + classifierNames[highestClf] + ', Score=' + str(highestScore)
print 'Feature Importances:'
print np.vstack((trainData.columns.values, classifiers[highestClf].feature_importances_)).T

# Make predictions

testData = cleanData('test.csv')
testX = testData.values.astype(int)
output = np.column_stack((testX[:,0], classifiers[highestClf].predict(testX)))
results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])

# print results
