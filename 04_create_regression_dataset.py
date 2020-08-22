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
   
ml_dataset = ml_dataset.drop(['episodes_watched', 'perc_watched'], axis = 1)

ml_dataset['season_num'] = pd.to_numeric([*map(lambda x: x.split('Season', -1)[1] if ('Season' in x)\
                                               else 1, ml_dataset['Season'])])

ml_dataset .to_pickle(directory + 'ml_data.pkl')