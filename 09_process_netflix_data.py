#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:45:40 2020

@author: bryanchia
"""

import pandas as pd
from datetime import datetime as dt
import numpy as np
import string 
import pickle5 as pickle

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/'

#ML Dataset

ml_vars = pd.read_pickle(directory + 'selected_ml_data.pkl')

watched_list = ml_vars['Title'].unique()

x_vars= ml_vars.iloc[:, 3:].columns

genre_vars = [str.replace(x, 'Genre_', '') for x in x_vars if 'Genre' in x]

tag_vars = [str.replace(x, 'Tag_', '') for x in x_vars if 'Tag' in x]

#Merge on characterisitcs

with open(directory + 'show_chars_full.pkl', 'rb') as f:
    df_vars = pickle.load(f)

df_vars = df_vars[~df_vars['Title'].isin(ml_vars['Title'])]

#Create list of characteristics

df_genre = df_vars[['Title', 'Genre']]

df_genre = df_genre[df_genre['Title'] != 'American Crime'].reset_index(drop = True)

s = pd.DataFrame([pd.Series(x) for x in df_genre['Genre']]).stack().reset_index(level = 1, drop = True)

s.name = 'Genres'

df_genre_long = df_genre.join(s, how = 'right').drop('Genre', axis = 1).reset_index(drop = True)

df_genre_long = df_genre_long[df_genre_long['Genres'].str.contains('20|19') == False]

df_genre_long = df_genre_long[df_genre_long['Genres'].isin(genre_vars)]

df_genre_long.to_pickle(directory + 'netflix_data/show_genres_stack_full.pkl')

#Create list of tags

with open(directory + "show_tags_full.pkl", "rb") as f:
    df_tags = pickle.load(f)
    

df_tags = df_tags[df_tags['Title'] != 'American Crime'].reset_index(drop = True)

s = pd.DataFrame([pd.Series(x)for x in df_tags['Tags']]).stack().reset_index(level = 1, drop = True) 

s.name = 'Tag'

df_tags_long = df_tags.join(s, how =  'right' ).drop('Tags', axis = 1).reset_index(drop = True)

df_tags_long = df_tags_long[~df_tags_long['Title'].isin(ml_vars['Title'])]

df_tags_long = df_tags_long[df_tags_long['Tag'].isin(tag_vars)]

df_tags_long.to_pickle(directory + 'netflix_data/show_tags_stack_full.pkl')

#Mimic ML Datasets

df_seasons = df_vars[['Title', 'Total Seasons']]

index = pd.DataFrame(range(1, max([int(x) for x in df_seasons['Total Seasons']])), columns = ['season_num'])

index['key'] = 0
df_seasons['key'] = 0

df_index = pd.merge(df_seasons, index, on = 'key', how = 'outer').drop('key', axis = 1)

df_index['Total Seasons'] = df_index['Total Seasons'].apply(int)

df_index = df_index[df_index['season_num'] <= df_index['Total Seasons']]

genre_dummies = pd.get_dummies(df_genre_long, prefix = 'Genre', columns = ['Genres']).groupby('Title').sum()

tag_dummies = pd.get_dummies(df_tags_long, prefix = 'Tag', columns = ['Tag']).groupby('Title').sum()

X_vars_fp = pd.merge(pd.merge(df_index, genre_dummies, how = "left", left_on = 'Title', right_on = 'Title'), \
                     tag_dummies, how = "left", left_on = 'Title', right_on = 'Title')
    
X_vars_fp.iloc[2:] = X_vars_fp.iloc[2:].reindex(columns = x_vars)