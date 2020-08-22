#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:21:22 2020

@author: bryanchia
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/'

ml = pd.read_pickle(directory + 'ml_data.pkl')

Y = ml.iloc[:, 4]
X = pd.merge(ml.iloc[:, 2:3], ml.iloc[:, 5:], how = "left", left_index = True, right_index= True)

X['episode_length'] =  X['episode_length'].fillna(X['episode_length'].mean()) 

X.iloc[:, 1:] = X.iloc[:, 1:].fillna(0) 

#Split the rows
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
   train_test_split(X, Y, test_size = 0.2, random_state = 1234, stratify = Y)
    
#Define the keras model    

model = Sequential()

model.add(Dense(500,
                input_shape = (2319,),
                activation = 'relu',
                kernel_initializer = 'RandomNormal')
          )

model.add(Dropout(0.5))

model.add(Dense(500,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dropout(0.5))


model.add(Dense(100,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dropout(0.5))

model.add(Dense(50,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dropout(0.5))

model.add(Dense(24,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dropout(0.5))

model.add(Dense(1,
                activation = 'sigmoid'))

#Compile the model

from keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch, lr):
   if (epoch > 50) & (epoch <= 100):
       lr = 1e-5
       return lr
   elif epoch > 100:
        lr = 1e-7
   return lr

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
              )

from keras import callbacks 

earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                        mode ="min", patience = 20,
                                        restore_best_weights = True) 

history = model.fit(X_train, Y_train, validation_data = [X_test, Y_test], epochs = 200, batch_size = 32,\
                    callbacks =[earlystopping, LearningRateScheduler(lr_scheduler, verbose=1)])

accuracy_test = model.evaluate(X_test, Y_test)

#Display model history

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Get the predicted values and predicted probabilities of Y test
Y_pred_prob = model.predict(X_test)
Y_predict = pd.Series([1 if x >0.5 else 0 for x in Y_pred_prob])

Y_predictions = pd.merge(pd.merge(pd.merge(ml[['Title', 'Season']], Y_test, how = "right", left_index = True, right_index = True).reset_index(drop = True),\
                         pd.DataFrame(Y_pred_prob, columns = ['percentage']), how = "right", left_index = True, right_index = True),\
                         pd.DataFrame(Y_predict, columns = ['prediction']), how = "right", left_index = True, right_index = True)

#Build the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, Y_predict)

