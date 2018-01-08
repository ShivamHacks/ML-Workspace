# Simple log function that prints and writes to text file
def record_print(string):
	print(string)
	with open('summary.txt', 'a') as file:
		file.write(string)

import numpy as np
import pandas as pd

store = pd.HDFStore('store.h5')

# Only need to run once
#df_train = pd.read_csv('train.csv')
#store['df_train'] = df_train

df_train = store['df_train']

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