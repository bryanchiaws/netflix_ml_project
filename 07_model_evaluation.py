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


directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/'

#Select variables
ms_results = pd.read_pickle(directory + 'model_selection.pkl')

df_results = pd.merge(pd.DataFrame(ms_results['params']), pd.DataFrame([pd.Series(x) for x in ms_results['results']]), left_index = True, right_index = True)

max_params = list(df_results.sort_values([0, 3]).iloc[0].loc['params'])

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

def train_model(X, Y):
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import callbacks 
    from keras.callbacks import LearningRateScheduler
    from tensorflow.keras import regularizers
    
    nodes1, nodes2, dp1, dp2, reg1, reg2 = max_params

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
    
    train_loss_vec = []
    train_acc_vec = []
    test_loss_vec = []
    test_acc_vec = []
    
    for i in range (0, 10):
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234 + i, stratify = Y)
        
        
        model.fit(X_train, Y_train, validation_data = [X_test, Y_test], epochs = 200, batch_size = 32,\
                    callbacks =[earlystopping, LearningRateScheduler(lr_scheduler, verbose=0)])
            
        train = model.evaluate(X_train, Y_train)
        test = model.evaluate(X_test, Y_test)
        
        test_loss_vec.append(test[0])
        test_acc_vec.append(test[1])
        
        train_loss_vec.append(train[0])
        train_acc_vec.append(train[1])
    
    return sum(test_loss_vec)/len(test_loss_vec), sum(test_acc_vec)/len(test_acc_vec), sum(train_loss_vec)/len(train_loss_vec), sum(train_acc_vec)/len(train_acc_vec)#, sum(rvec)/len(rvec)

#ml_vars = shuffle(pd.read_pickle(directory + 'selected_ml_data.pkl'), random_state=0)

ml_vars = pd.read_pickle(directory + 'selected_ml_data.pkl')

train_loss_vec = []
train_acc_vec = []
test_loss_vec = []
test_acc_vec = []

#random.seed(30)

for i in range(11, len(ml_vars)+1, 5):
    
    randlist= [random.randint(0, len(ml_vars)-1) for x in range(0, i)]
    
    ml_vars_ss = ml_vars.iloc[randlist]
    
    #ml_vars_ss = ml_vars.head(i)

    Y = ml_vars_ss.iloc[:, 2]
    X = ml_vars_ss.iloc[:, 3:]

    X['episode_length'] =  X['episode_length'].fillna(X['episode_length'].mean()) 

    X.iloc[:, 0:] = X.iloc[:, 0:].fillna(0)
    
    test_loss, test_acc, train_loss, train_acc = train_model(X, Y)
    
    test_loss_vec.append(test_loss)
    test_acc_vec.append(test_acc)
    train_loss_vec.append(train_loss)
    train_acc_vec.append(train_acc)
    
# summarize history for loss

test_loss_df = pd.Series(test_loss_vec, index = range(11, len(ml_vars)+1, 5))
train_loss_df = pd.Series(train_loss_vec, index = range(11, len(ml_vars)+1, 5))

plt.plot(test_loss_df)
plt.plot(train_loss_df)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('sample size')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#summarize history for accuracy

test_acc_df = pd.Series(test_acc_vec, index = range(11, len(ml_vars)+1, 5))
train_acc_df = pd.Series(train_acc_vec, index = range(11, len(ml_vars)+1, 5))

plt.plot(test_acc_df)
plt.plot(train_acc_df)
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('sample size')
plt.legend(['train', 'test'], loc='upper right')
plt.show()