import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
All State Claims Severity:

Problem Type: Regression
Features: Labels and Continuous
"""

# Obtain Data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_complete = df_train.drop(['id', 'loss'], axis=1).append(df_test.drop(['id'], axis=1), ignore_index=True)

"""
store = pd.HDFStore('store.h5')
X_train = store['X_train']
X_test = store['X_test']
y_train = store['y_train']
"""

# Graph Histogram of Output
def histogram():
	plt.clf() # clear figure
	plt.xlabel('Loss')
	plt.ylabel('Count')
	plt.hist(df_train['loss'], bins=30)
	plt.show()

# List Features and Label Frequences
def feature_analysis():
	open('summary.txt', 'w').close() # clear file
	with open('summary.txt', 'a') as summary:
		features = df_complete.columns.values
		summary.write('Features: ' + str(features) + '\n')
		summary.write('\nFeatures Counts:')
		for feature in features:
			summary.write('\n---------------\n')
			if (df_complete[feature].dtype == 'object'):
				summary.write(str(df_complete[feature].value_counts()))
			else: # numeric data
				summary.write(str(df_complete[feature].describe()))

feature_analysis()
histogram()

# Summary of Data
"""
open('summary.txt', 'w').close() # clear file
record_print(df_train.info().to_string())
record_print('----------------------------------------')
record_print(df_train.describe(include='all').to_string())
record_print('----------------------------------------')
record_print(df_train.head(10).to_string())
record_print('----------------------------------------')
"""