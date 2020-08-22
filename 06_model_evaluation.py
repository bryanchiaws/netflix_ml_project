#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 20:29:05 2020

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

