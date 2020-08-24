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
from tensorflow.keras import regularizers

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/'

ml_vars = pd.read_pickle(directory + 'selected_ml_data.pkl')

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

Y_predictions = pd.merge(pd.merge(pd.merge(ml_vars[['Title', 'Season']], Y_test, how = "right", left_index = True, right_index = True).reset_index(drop = True),\
                         pd.DataFrame(Y_pred_prob, columns = ['percentage']), how = "right", left_index = True, right_index = True),\
                         pd.DataFrame(Y_predict, columns = ['prediction']), how = "right", left_index = True, right_index = True)

#Build the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, Y_predict)

p = cm[0,0]/(cm[0,0] + cm[0, 1])
r = cm[0,0]/(cm[0,0] + cm[1, 0])

#Cross validation

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
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234 + i*10, stratify = Y)
        
        
        model.fit(X_train, Y_train, validation_data = [X_test, Y_test], epochs = 200, batch_size = 32,\
                    callbacks =[earlystopping, LearningRateScheduler(lr_scheduler, verbose=1)])

        test = model.evaluate(X_test, Y_test)
        
        
        Y_predict = model.predict_classes(X_test)
        cm = confusion_matrix(Y_test, Y_predict)

        pvec.append(cm[0,0]/(cm[0,0] + cm[0, 1]))
        rvec.append(cm[0,0]/(cm[0,0] + cm[1, 0]))
        
        loss_vec.append(test[0])
        acc_vec.append(test[1])
        
    
    return loss_vec, acc_vec

test = train_model([50,50, 0.2, 0.2, 1e-4, 1e-6])

mean_vals = [sum(x)/len(x) for x in test]