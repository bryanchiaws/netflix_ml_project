#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:21:22 2020

@author: bryanchia
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

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

model.add(Dense(1400,
                input_shape = (2319,),
                activation = 'relu',
                kernel_initializer = 'RandomNormal')
          )


model.add(Dense(450,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(225,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(100,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(50,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(20,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(10,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(1,
                activation = 'sigmoid'))

#Compile the model

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
              )

model.fit(X_train, Y_train, epochs = 500, batch_size = 32)

accuracy_test = model.evaluate(X_test, Y_test)

#Get the predicted values and predicted probabilities of Y test
Y_predict = model.predict_classes(X_test)
Y_pred_prob = model.predict(X_test)

Y_predictions = pd.merge(pd.merge(pd.merge(ml[['Title', 'Season']], Y_test, how = "right", left_index = True, right_index = True).reset_index(drop = True),\
                         pd.DataFrame(Y_pred_prob, columns = ['percentage']), how = "right", left_index = True, right_index = True),\
                         pd.DataFrame(Y_predict, columns = ['prediction']), how = "right", left_index = True, right_index = True)

#Build the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, Y_predict)

