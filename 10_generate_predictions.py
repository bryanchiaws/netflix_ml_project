#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 22:15:15 2020

@author: bryanchia
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.metrics import confusion_matrix
import random
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
import pickle5 as pickle

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/'

ml_vars = pd.read_pickle(directory + 'bryan_viewing/selected_ml_data.pkl')

Y = ml_vars.iloc[:, 2]
X = ml_vars.iloc[:, 3:]

X['episode_length'] =  X['episode_length'].fillna(X['episode_length'].mean()) 

X.iloc[:, 0:] = X.iloc[:, 0:].fillna(0) 

#Split the rows
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
   train_test_split(X, Y, test_size = 0.2, random_state = 1234, stratify = Y)
    
#Define the keras model    

model = Sequential()

model.add(Dense(80,
                input_shape = (188,),
                activation = 'relu',
                kernel_initializer = 'RandomNormal',
                activity_regularizer = regularizers.l2(1e-4))
          )

model.add(Dropout(0.2))

model.add(Dense(80,
                activation = 'relu',
                kernel_initializer = 'RandomNormal',
                activity_regularizer = regularizers.l2(1e-6))
          )

model.add(Dropout(0.2))

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
                                        mode ="min", patience = 30,
                                        restore_best_weights = True,
                                        verbose = 1) 

history = model.fit(X_train, Y_train, validation_data = [X_test, Y_test], epochs = 200, batch_size = 32,\
                    callbacks =[earlystopping, LearningRateScheduler(lr_scheduler, verbose=1)])

accuracy_test = model.evaluate(X_test, Y_test)


#Bring in prediction possibilities

with open(directory + 'netflix_data/full_potential_predictions.pkl', 'rb') as f:
    potentials = pickle.load(f)

#Get the predicted values and predicted probabilities of Y test

potentials['episode_length'] =  potentials['episode_length'].fillna(potentials['episode_length'].mean()) 

potentials.iloc[:, 2:] = potentials.iloc[:, 2:].fillna(0) 

Y_pred_prob = model.predict(potentials.iloc[:,1:])
Y_predict = pd.Series([1 if x >0.5 else 0 for x in Y_pred_prob])

Y_predictions = pd.merge(pd.merge(potentials[['Title', 'season_num']], 
                         pd.DataFrame(Y_pred_prob, columns = ['percentage']), how = "right", left_index = True, right_index = True),\
                         pd.DataFrame(Y_predict, columns = ['prediction']), how = "right", left_index = True, right_index = True)
    
final_predictions = Y_predictions[Y_predictions['season_num'] == 1 ].drop_duplicates().sort_values('percentage', ascending = False)

final_predictions.to_excel(directory +  'final_predictions.xlsx')
