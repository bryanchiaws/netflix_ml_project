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

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data/bryan_viewing/'

with open(directory + 'show_data.pkl', 'rb') as f:
    df_vd = pickle.load(f)
    
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

#Remove some shows I didn't watch
df_vd = df_vd[df_vd['Show'] != 'I Am Not Okay With This']
df_vd = df_vd[df_vd['Show'] != 'Dead to Me']

#Clean up some shows
df_vd['Show'] = ['American Crime Story' if x == 'American Crime' else x for x in df_vd['Show']]

#Merge on characterisitcs

with open(directory + 'show_chars_full.pkl', 'rb') as f:
    df_vars = pickle.load(f)

df_chars = pd.merge(df_vd, df_vars, how = "left", left_on = 'Show', right_on = 'Title')

df_chars['Length_Minutes'] = [convert_minutes(x) for x in df_chars['Length']]


df_chars['Episodes'] = ['12 episodes' if ('Kingdom' in x) else y for x, y in zip(df_chars['Show'], df_chars['Episodes'])]
    

df_chars['Num_Episodes'] = pd.to_numeric([str(x).split('episodes', -1)[0] if str(x).find('episodes') != -1\
                                          else 1 for x in df_chars['Episodes']])

df_chars.loc[df_chars['Season'].isnull(), 'Season'] = 'Season 1'

#df_chars['Season'] = [ 'Season 1' if x == '' else x for x in df_chars['Season']]

 #Account for mistake for American Crime Story
df_chars['Season'] = [*map(lambda x, y: 'Season 2' if ('Gianni' in x)\
                                 else y, df_chars['Original Title'], df_chars['Season'])]
    
#Account for mistake for some limited series
df_chars['Length_Minutes'] = [*map(lambda x, y, z, a: y/z if ('Limited' in str(x)) & ('Unbelievable' not in str(a))\
                                 else y, df_chars['Full Title'], df_chars['Length_Minutes'], df_chars['Num_Episodes'],df_chars['Title'])]

    
df_chars['Season_Num'] = pd.to_numeric([*map(lambda x: x.split('Season', -1)[1] if ('Season' in x)\
                                               else 1, df_chars['Season'])])

df_chars['Total Time'] = [*map(lambda x, y: x * y, df_chars['Num_Episodes'], df_chars['Length_Minutes'])]

df_chars['Total_Seasons'] = pd.to_numeric([*map(lambda x: str(x).split('.E', -1)[0].split('S', -1)[1] if ('S' in str(x))\
                                               else x, df_chars['Total Seasons'])])
    
#Account for mistake for American Crime Story
df_chars['Total_Seasons'] = [*map(lambda x, y: 3 if ('American Crime Story' in x)\
                                 else y, df_chars['Show'], df_chars['Total_Seasons'])]
    
#Account for mistake for Derry Girls
df_chars['Total_Seasons'] = [*map(lambda x, y: 2 if ('Derry' in x)\
                                 else y, df_chars['Show'], df_chars['Total_Seasons'])]

    
df_chars['Eps_Per_Season'] = pd.to_numeric([*map(lambda x,y: round(x/y, 0), df_chars['Num_Episodes'], df_chars['Total_Seasons'])])

#Filter to TV Shows

df_chars = df_chars[df_chars['Num_Episodes'] > 1]

#Retention Analysis

df_sum_retention = df_chars.groupby(['Title', 'Season']).agg(
     episode_length = ('Length_Minutes', max),
     episodes_watched = ('Episode', 'nunique'),
     num_episodes = ('Eps_Per_Season', max)
     ).reset_index()

df_sum_retention['perc_watched'] = [*map(lambda x, y: x/y, df_sum_retention['episodes_watched'], df_sum_retention['num_episodes'] )]

df_sum_retention['finished'] = [1 if x >=1 else 0 for x in df_sum_retention['perc_watched']]

df_sum_retention = df_sum_retention[(df_sum_retention['num_episodes'] > 1)].sort_values(['perc_watched'], ascending = False)

df_sum_retention.to_csv(directory + 'retention_summary_to_edit.csv')

df_sum_retention_manual_edits = pd.read_csv(directory + 'retention_summary_edited.csv', index_col = 0)

with open(directory + "retention_stats.pkl", "wb") as f:
    pickle.dump(df_sum_retention_manual_edits, f)

#Create list of characteristics

df_genre = df_vars[['Title', 'Genre']]

df_genre = df_genre[df_genre['Title'] != 'American Crime'].reset_index(drop = True)

s = pd.DataFrame([pd.Series(x) for x in df_genre['Genre']]).stack().reset_index(level = 1, drop = True)

s.name = 'Genres'

df_genre_long = df_genre.join(s, how = 'right').drop('Genre', axis = 1).reset_index(drop = True)

df_genre_long = df_genre_long[df_genre_long['Genres'].str.contains('20|19') == False]

df_genre_long = df_genre_long[df_genre_long['Title'].isin(df_sum_retention['Title'])]

df_genre_long.to_pickle(directory + 'show_genres_stack_full.pkl')

df_sum_genre = df_genre_long.groupby('Genres').agg(
    count = ('Genres', 'count'),
    shows = ('Title', lambda x : ', '.join(x))
    ).reset_index()

#Create list of actors

with open(directory + 'show_actors_full.pkl', 'rb') as f:
    df_actors  = pickle.load(f)

df_actors = df_actors[df_actors['Title'] != 'American Crime'].reset_index(drop = True)

s = pd.DataFrame([pd.Series(x)for x in df_actors['Actors']]).stack().reset_index(level = 1, drop = True) 

s.name = 'Actor'

df_actors_long = df_actors.join(s, how =  'right' ).drop('Actors', axis = 1).reset_index(drop = True)

with open(directory + "show_actors_stack_full.pkl", "wb") as f:
    pickle.dump(df_actors_long, f)

df_sum_actors = df_actors_long.groupby('Actor').agg(
    count = ('Actor', 'count'),
    shows = ('Title', lambda x : ', '.join(x))
    ).reset_index()

#Create list of tags

with open(directory + "show_tags_full.pkl", "rb") as f:
    df_tags = pickle.load(f)

df_tags = df_tags[df_tags['Title'] != 'American Crime'].reset_index(drop = True)

s = pd.DataFrame([pd.Series(x)for x in df_tags['Tags']]).stack().reset_index(level = 1, drop = True) 

s.name = 'Tag'

df_tags_long = df_tags.join(s, how =  'right' ).drop('Tags', axis = 1).reset_index(drop = True)

df_tags_long = df_tags_long[(df_tags_long['Tag'] != 'bare chested male') & (df_tags_long['Tag'] != '')]

df_tags_long = df_tags_long[df_tags_long['Title'].isin(df_sum_retention['Title'])]

df_tags_long.to_pickle(directory + 'show_tags_stack_full.pkl')

df_sum_tags = df_tags_long.groupby('Tag').agg(
    count = ('Tag', 'count'),
    shows = ('Title', lambda x : ', '.join(x))
    ).reset_index()