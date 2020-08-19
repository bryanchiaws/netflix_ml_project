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
 
diabetes = pd.read_csv('diabetes.csv')

diabetes.isnull().sum(axis = 0)

X = diabetes.iloc[:, 0:-1]
Y = diabetes.iloc[:, -1]

#Split the rows
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size = 0.2, random_state = 1234, stratify = Y)
    
#Define the keras model    

model = Sequential()

model.add(Dense(24,
                input_shape = (8,),
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(12,
                activation = 'relu',
                kernel_initializer = 'RandomNormal'))

model.add(Dense(1,
                activation = 'sigmoid'))

#Compile the model

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
              )

model.fit(X_train, Y_train, epochs = 160, batch_size = 10)

accuracy_test = model.evaluate(X_test, Y_test)

#Get the predicted values and predicted probabilities of Y test
Y_predict = model.predict_classes(X_test)
Y_pred_prob = model.predict(X_test)

#Build the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, Y_predict)

