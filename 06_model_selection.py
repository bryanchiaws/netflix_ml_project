#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 20:29:05 2020

@author: bryanchia
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from itertools import product
import multiprocessing as mp
from sklearn.metrics import confusion_matrix

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/'

ml_vars = pd.read_pickle(directory + 'selected_ml_data.pkl')

Y = ml_vars.iloc[:, 2]
X = ml_vars.iloc[:, 3:]

X['episode_length'] =  X['episode_length'].fillna(X['episode_length'].mean()) 

X.iloc[:, 0:] = X.iloc[:, 0:].fillna(0) 

#Split the rows
from sklearn.model_selection import train_test_split
   
#Learning rate scheduler


def lr_scheduler(epoch, lr):
   if (epoch > 50) & (epoch <= 100):
       lr = 1e-5
       return lr
   elif epoch > 100:
        lr = 1e-7
   return lr


def train_model(params):
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import callbacks 
    from keras.callbacks import LearningRateScheduler
    from tensorflow.keras import regularizers
    
    nodes1, nodes2, dp1, dp2, reg1, reg2 = params

    model = Sequential()
    
    model.add(Dense(nodes1,
                input_shape = (188,),
                activation = 'relu',
                kernel_initializer = 'RandomNormal',
                activity_regularizer = regularizers.l2(reg1))
          )

    model.add(Dropout(dp1))

    model.add(Dense(nodes2,
                activation = 'relu',
                kernel_initializer = 'RandomNormal',
                activity_regularizer = regularizers.l2(reg2))
          )

    model.add(Dropout(dp2))

    model.add(Dense(1,
                activation = 'sigmoid'))
    
    model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy']
              )
    
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  
                                        mode ="min", patience = 30,
                                        restore_best_weights = True) 
    
    loss_vec = []
    acc_vec = []
    pvec = []
    rvec = []
    
    for i in range (0, 10):
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234 + i, stratify = Y)
        
        
        model.fit(X_train, Y_train, validation_data = [X_test, Y_test], epochs = 200, batch_size = 32,\
                    callbacks =[earlystopping, LearningRateScheduler(lr_scheduler, verbose=1)])

        test = model.evaluate(X_test, Y_test)
        
        
        Y_predict = model.predict_classes(X_test)
        cm = confusion_matrix(Y_test, Y_predict)

        pvec.append(cm[0,0]/(cm[0,0] + cm[0, 1]))
        rvec.append(cm[0,0]/(cm[0,0] + cm[1, 0]))
        
        loss_vec.append(test[0])
        acc_vec.append(test[1])
        
    
    return [sum(loss_vec)/len(loss_vec), sum(acc_vec)/len(acc_vec), sum(pvec)/len(pvec), sum(rvec)/len(rvec)]


nodes1 = np.arange(12, 61, 24)
nodes2 = np.arange(12, 61, 24)
dp1 = np.arange(0.2, 0.81, 0.3)
dp2 = np.arange(0.2, 0.81, 0.3)
reg1 = np.array([1e-4, 1e-5, 1e-6])
reg2 = np.array([1e-4, 1e-5, 1e-6])

params = list(product(nodes1, nodes2, dp1, dp2, reg1, reg2))

pool = mp.Pool(processes = 12)

#import time

#start = time.time()
test = pool.map(train_model, params)
#end = time.time()

#print(f"Runtime of the program is {end - start}")
    
results = pd.DataFrame(zip(params, test), columns = ['params', 'results'])

results.to_pickle(directory + 'model_selection.pkl')
