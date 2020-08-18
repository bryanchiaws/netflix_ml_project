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
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import PercentFormatter

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/clean_data'

df_chars = pd.read_pickle(directory + '/show_data_full.pkl')

def convert_minutes(x):
    
    x = str(x)
    
    if x == '':
        return np.NaN
    elif ('h' in x) & ('min' not in x):
        return pd.to_numeric(x.split('h', -1)[0])*60.0
    elif ('h' not in x) & ('min' in x):
        return pd.to_numeric(x.split('min', -1)[0])*1.0
    elif ('h' in x) & ('min' in x):
        return pd.to_numeric(x.split('h', -1)[0])*60.0 + pd.to_numeric(x.split('h', -1)[1].split('min', -1)[0])

#Remove some shows I didn't watch
df_chars = df_chars[df_chars['Show'] != 'I Am Not Okay With This']
df_chars = df_chars[df_chars['Show'] != 'Dead to Me']
#df_chars = df_chars[df_chars['Show'] != 'The Inbetweeners']


df_chars['Length_Minutes'] = df_chars['Length'].apply(convert_minutes)
    
#Account for mistake for Kingdom
df_chars['Episodes'] = [*map(lambda x, y: '12 episodes' if ('Kingdom' in x)\
                                 else y, df_chars['Show'], df_chars['Episodes'])]

df_chars['Num_Episodes'] = pd.to_numeric([*map(lambda x: str(x).split('episodes', -1)[0] if str(x).find('episodes') != -1\
                                 else 1, df_chars['Episodes'])])
    
df_chars.loc[df_chars['Season'].isnull(), 'Season'] = 'Season 1'
    
#Account for mistake for s0me limited series
df_chars['Length_Minutes'] = [*map(lambda x, y, z, a: y/z if ('Limited' in str(x)) & ('Unbelievable' not in str(a))\
                                 else y, df_chars['Full Title'], df_chars['Length_Minutes'], df_chars['Num_Episodes'],df_chars['Title'])]

    
df_chars['Season_Num'] = pd.to_numeric([*map(lambda x: x.split('Season', -1)[1] if ('Season' in x)\
                                               else 1, df_chars['Season'])])

df_chars['Total Time'] = [*map(lambda x, y: x * y, df_chars['Num_Episodes'], df_chars['Length_Minutes'])]

df_chars['Total_Seasons'] = pd.to_numeric([*map(lambda x: str(x).split('.E', -1)[0].split('S', -1)[1] if ('S' in str(x))\
                                               else x, df_chars['Total Seasons'])])

#Retention Analysis

df_sum_retention = df_chars.groupby(['Title']).agg(
     time_spent = ('Length_Minutes', sum),
     episode_length = ('Length_Minutes', max),
     time_spent_h = ('Length_Minutes', lambda x: sum(x)/60.0),
     total_time = ('Total Time', min),
     total_time_h = ('Total Time', lambda x: min(x)/60.0),
     date_range = ('Date', lambda x: (max(x) - min(x)).days),
     total_seasons= ('Total_Seasons', max),
     seasons_watched = ('Season_Num', 'nunique'),
     episodes_watched = ('Episode', 'nunique'),
     num_episodes = ('Num_Episodes', max)
     ).reset_index()

df_sum_retention['perc_watched'] = [*map(lambda x, y: x/y, df_sum_retention['time_spent'], df_sum_retention['total_time'] )]
