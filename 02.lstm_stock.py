# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import math
import numpy as np

import pandas_datareader as web 
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv2D
from tensorflow.keras import Model, Sequential
import tensorflow as tf

PERIOD_AS_INPUT = 60
CLOSE = "Close" # "Adj Close"
PREDICT_BASED_ON_RESULT = False 

#Get the stock quote
#df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-02-27')
df = web.DataReader('005930.KS', data_source='yahoo', start='2019-01-01', end='2020-02-28')

#Show teh data
#print(df)
print("df.shape:",df.shape)

def visualize_stock():
    #Visualize the closing price history
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df[CLOSE])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()

#visualize_stock()

#Create a new dataframe with only the 'Close column
#data = df.filter(items = [CLOSE, 'High', 'Low', 'Volume'])
data = df.filter(items = [CLOSE])

#Convert the dataframe to a numpy array
dataset = data.values
print("dataset:", dataset.shape)

#Get the number of rows to train the model on
training_data_len = math.ceil( len(dataset) * .8 )
test_data_len = len(dataset) - training_data_len

print("training_data_len: ",training_data_len)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print("scaled_data:",scaled_data.shape)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]

#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(PERIOD_AS_INPUT, len(train_data)):
  x_train.append(train_data[i-PERIOD_AS_INPUT:i, :])
  y_train.append(train_data[i, 0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print()

#Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, x_train.shape)
print("x_train.shape:", x_train.shape)

# Create a RNN model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= x_train.shape[-2:]))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - PERIOD_AS_INPUT: , :]

#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(PERIOD_AS_INPUT, len(test_data)):
  x_test.append(test_data[i-PERIOD_AS_INPUT:i, :])
  break

#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, x_test.shape)

print("x_test:", x_test.shape)
#Get the models predicted price values 

if PREDICT_BASED_ON_RESULT:
  predictions = model.predict(x_test)
else:
  predictions = []
  for _ in range(test_data_len):
      print(x_test)
      pre = model.predict(x_test)
      pre = np.reshape(pre, [1,1,1])
      print("pre:", pre.shape, pre)
      x_test = np.append(x_test[:,1:], pre)
      x_test = np.reshape(x_test, [1,PERIOD_AS_INPUT,1])
      predictions.append(pre[0][0][0])

  predictions = np.array(predictions)
  predictions = np.reshape(predictions, [test_data_len, 1])


print("predictions:", predictions.shape)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print("rmse:", rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train[CLOSE])
plt.plot(valid[[CLOSE, 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# end of file