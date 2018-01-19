import numpy as np
import pandas as pd



from keras.models import Sequential
from keras.layers import Dense, Dropout


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Obtain Data
store = pd.HDFStore('store.h5')

X_train = store['X_train']
X_test = store['X_test']
y_train = store['y_train']
test_ids = store['test_ids']

input_dim = len(X_train.iloc[0])

print('Obtained data')
print('Input Dim: ' + str(input_dim))

# Don't need these as validation_split handles it
#trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.75, random_state=0)
#print('Training/Test partition made')

def build_model():
	model = Sequential()
	model.add(Dense(units=500, activation='relu', input_dim=input_dim))
	model.add(Dropout(0.2))
	model.add(Dense(units=250, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(units=128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(units=1))
	model.compile(loss='mae', optimizer='adam')
	return model

model = build_model()
model.fit(X_train, y_train, validation_split=0.2, batch_size=100, nb_epoch=10, verbose=1)

predY = model.predict(X_test).reshape(len(test_ids), )

print('Made Predictions')

# Save Output
result = list(zip(test_ids.tolist(), predY))
df_submission = pd.DataFrame(result, columns=['id', 'loss'])
df_submission.to_csv('submission.csv', index=False)

print('Saved Submission')