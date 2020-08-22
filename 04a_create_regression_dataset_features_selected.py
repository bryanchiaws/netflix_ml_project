#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:45:40 2020

@author: bryanchia
"""

import pickle5 as pickle
import pandas as pd
from datetime import datetime as dt
import numpy as np
import string 

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/'

with open(directory + 'retention_stats.pkl', 'rb') as f:
    retention_metrics = pickle.load(f)

with open(directory + 'show_genres_stack_full.pkl', 'rb') as f:
    genres = pickle.load(f)

with open(directory + "show_tags_stack_full.pkl", "rb") as f:
    tags = pickle.load(f)
    
genre_dummies = pd.get_dummies(genres, prefix = 'Genre', columns = ['Genres']).groupby('Title').sum()

#Just use most popular tags (>=4 shows)

df_sum_tags = tags.groupby('Tag').agg(
    count = ('Tag', 'count'),
    shows = ('Title', lambda x : ', '.join(x))
    ).reset_index()

tags = tags[tags['Tag'].isin(df_sum_tags[df_sum_tags['count']>=1]['Tag'])]

tag_dummies = pd.get_dummies(tags, prefix = 'Tag', columns = ['Tag']).groupby('Title').sum()

ml_dataset = pd.merge(pd.merge(retention_metrics, genre_dummies, how = "left", left_on = 'Title', right_on = 'Title'),\
                       tag_dummies, how = "left", left_on = 'Title', right_on = 'Title')
   
ml = ml_dataset.drop(['episodes_watched', 'perc_watched'], axis = 1)

ml['season_num'] = pd.to_numeric([*map(lambda x: x.split('Season', -1)[1] if ('Season' in x)\
                                               else 1, ml_dataset['Season'])])
    
#Determine what features to use using correlation matrices 

X = pd.merge(ml.iloc[:, 2:3], ml.iloc[:, 5:], how = "left", left_index = True, right_index= True)

X_corr_i = X.corr().reset_index()

X_corr = X.corr()

columns = np.full((X_corr.shape[0],), True, dtype=bool)

#Keep only columns with correlation less than 0.9

for i in range(X_corr.shape[0]):
    for j in range(i+1, X_corr.shape[0]):
        if X_corr.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False
                
selected_X = X[X.columns[columns]]

Y = pd.merge(ml.iloc[:, 0:2], ml.iloc[:, 4], how = "left", left_index = True, right_index= True)

selected_ml = pd.merge(Y, selected_X, how = "right", left_index = True, right_index = True)

selected_ml.to_pickle(directory + 'selected_ml_data.pkl')