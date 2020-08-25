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

ml_vars = pd.read_pickle(directory + 'bryan_viewing/selected_ml_data.pkl')

watched_list = ml_vars['Title'].unique()

x_vars= ml_vars.iloc[:, 3:].columns

genre_vars = [x for x in x_vars if 'Genre' in x]

tag_vars = [x for x in x_vars if 'Tag' in x]

#Merge on characterisitcs

with open(directory + 'netflix_data/show_chars_full.pkl', 'rb') as f:
    df_vars = pickle.load(f)
    
df_vars['Total Seasons'] = [*map(lambda x: str(x).split('.E', -1)[0].split('S', -1)[1] if ('S' in str(x))\
                                               else x, df_vars['Total Seasons'])]
    
df_vars['Total Seasons'] = [1 if x == 'Unknown' else int(x) for x in df_vars['Total Seasons']]

#Create list of characteristics

df_genre = df_vars[['Title', 'Genre']]

df_genre = df_genre[df_genre['Title'] != 'American Crime'].reset_index(drop = True)

s = pd.DataFrame([pd.Series(x) for x in df_genre['Genre']]).stack().reset_index(level = 1, drop = True)

s.name = 'Genres'

df_genre_long = df_genre.join(s, how = 'right').drop('Genre', axis = 1).reset_index(drop = True)

df_genre_long = df_genre_long[df_genre_long['Genres'].str.contains('20|19') == False]

df_genre_long.to_pickle(directory + 'netflix_data/show_genres_stack_full.pkl')

genre_dummies = pd.get_dummies(df_genre_long, prefix = 'Genre', columns = ['Genres']).groupby('Title').sum()

#Filter only to genres in the data
genre_dummies = genre_dummies[genre_vars]

#Create list of tags

with open(directory + "netflix_data/show_tags_full.pkl", "rb") as f:
    df_tags = pickle.load(f)
    
df_tags = df_tags[df_tags['Title'] != 'American Crime'].reset_index(drop = True)

s = pd.DataFrame([pd.Series(x)for x in df_tags['Tags']]).stack().reset_index(level = 1, drop = True) 

s.name = 'Tag'

df_tags_long = df_tags.join(s, how =  'right' ).drop('Tags', axis = 1).reset_index(drop = True)

df_tags_long.to_pickle(directory + 'netflix_data/show_tags_stack_full.pkl')

tag_dummies = pd.get_dummies(df_tags_long, prefix = 'Tag', columns = ['Tag']).groupby('Title').sum()

tag_dummies['Tag_cakes'] = 0
tag_dummies['Tag_financial ruin'] = 0
tag_dummies['Tag_catfishing'] = 0

#Filter only to tags in the data
tag_dummies = tag_dummies[tag_vars]

#Mimic ML Datasets

def convert_minutes(x):
    
    x = str(x)
    
    if x == '':
        return np.nan
    elif ('h' in x) & ('min' not in x):
        return pd.to_numeric(x.split('h', -1)[0])*60.0
    elif ('h' not in x) & ('min' in x):
        return pd.to_numeric(x.split('min', -1)[0])*1.0
    elif ('h' in x) & ('min' in x):
        return pd.to_numeric(x.split('h', -1)[0])*60.0 + pd.to_numeric(x.split('h', -1)[1].split('min', -1)[0])

df_vars['episode_length'] = [convert_minutes(x) for x in df_vars['Length']]

df_vars['num_episodes'] = pd.to_numeric([str(x).split('episodes', -1)[0] if str(x).find('episodes') != -1\
                                          else 1 for x in df_vars['Episodes']])

#Account for some mistakes in episode length
df_vars['episode_length'] = [x/y if x > 130 else x for x, y in zip(df_vars['episode_length'], df_vars['num_episodes'])]
    
df_seasons = df_vars[['Title', 'Total Seasons', 'episode_length']]

index = pd.DataFrame(range(1, max([int(x) for x in df_seasons['Total Seasons']])), columns = ['season_num'])

index['key'] = 0
df_seasons['key'] = 0

df_index = pd.merge(df_seasons, index, on = 'key', how = 'outer').drop('key', axis = 1)

df_index['Total Seasons'] = df_index['Total Seasons'].apply(int)

df_index = df_index[df_index['season_num'] <= df_index['Total Seasons']]

X_vars_fp = pd.merge(pd.merge(df_index, genre_dummies, how = "left", left_on = 'Title', right_on = 'Title'), \
                     tag_dummies, how = "left", left_on = 'Title', right_on = 'Title')
    
#Filter out what I have already watched
potential_predictors  = X_vars_fp[~ X_vars_fp['Title'].str.lower().isin(ml_vars['Title'].str.lower())]

#Will definitely not watch animation
potential_predictors  = potential_predictors[potential_predictors['Genre_Animation'] == 0]

pp_final = pd.merge(potential_predictors['Title'], \
                    potential_predictors[x_vars], how = "right", left_index = True, right_index = True).reset_index(drop = True)
    
pp_final.to_pickle(directory + 'netflix_data/full_potential_predictions.pkl')
