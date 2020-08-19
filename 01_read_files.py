#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:09:58 2020

@author: bryanchia
"""

import pandas as pd
from datetime import datetime as dt
import re

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/'

file = 'raw_data/bryan_viewing.csv'

df = pd.read_csv(directory + file)

df['Show'] = df['Title'].str.split(':', -1, True)[0]
df['Season'] = df['Title'].str.split(':', -1, True)[1]
df['Episode'] = df['Title'].str.split(':', -1, True)[2]

df['Date'] = [re.sub('/201', '/1', x) for x in df['Date']]

df['Date'] = [ dt.strptime(x, '%d/%m/%y') if (y == 'raena') else (dt.strptime(x, '%d/%m/%y') if (y == 'new') else dt.strptime(x, '%m/%d/%y')) for x, y in \
              zip(df['Date'], df['Dataset'])]

df = df.rename({'Title': 'Full Title'}, axis=1)

df.to_pickle(directory + 'clean_data/show_data.pkl')


