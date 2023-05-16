import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

data = pd.read_csv('finaldata.csv')
targets = data.drop(['acc_X', 'acc_Y', 'acc_Z', 'gyro_X', 'gyro_Y', 'gyro_Z', 'prox'], 1)
#print (targets)
#targets = targets.to_numpy()
data = data.drop('label', 1)
dataset = tf.keras.utils.timeseries_dataset_from_array(data,targets, sequence_length=10, sequence_stride=10)
print (dataset)
#dataset = dataset.batch(128,drop_remainder=True)
#print (dataset)
#for i in dataset:
#    print (i)
train_set, test_set = tf.keras.utils.split_dataset(dataset, left_size=0.8)
model = Sequential()
model.add(Bidirectional(LSTM(32, dropout=0.25, return_sequences=True)))
#model.add(Bidirectional(LSTM(10, dropout=0.25)))
model.add(Bidirectional(LSTM(32, dropout=0.25, return_sequences=True)))
model.add(Bidirectional(LSTM(32, dropout=0.25, return_sequences=True)))
model.add(Bidirectional(LSTM(32, dropout=0.25, return_sequences=True)))
model.add(Bidirectional(LSTM(32, dropout=0.25, return_sequences=True)))
model.add(Bidirectional(LSTM(32, dropout=0.25, return_sequences=True)))
model.add(Bidirectional(LSTM(32, dropout=0.25)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
model.fit(train_set, epochs=1, verbose=2)
loss, accuracy = model.evaluate(test_set)
z = pd.read_csv('~/dataaa/Prasanth/test.csv')
dataset_p = tf.keras.utils.timeseries_dataset_from_array(z, targets=None, sequence_length=10, sequence_stride=10)
model.predict[z.to_numpy()]
print('Accuracy: %.2f' %(accuracy*100))
print('Loss: %.2f' %(loss*100))