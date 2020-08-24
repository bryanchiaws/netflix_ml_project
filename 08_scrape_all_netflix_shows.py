# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from selenium import webdriver
import re
from selenium.webdriver.common.keys import Keys
import sys
import time
import csv
import xlwts
import string
import numpy as np
import pandas as pd
from datetime import datetime as dt

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/'

#Call chromedriver
chromedriver = directory + '/chromedriver'
os.environ['webdriver.chrome.driver'] = chromedriver
driver = webdriver.Chrome(chromedriver)

show_title = []       

driver.implicitly_wait(0.5)

driver.get('https://reelgood.com/tv/source/netflix')
    
time.sleep(2)

for s in range(0, 1801, 50):
    for i in range (s+1, s+51):
        print(i)
        try:
            show_name = driver.find_element_by_css_selector('#app_mountpoint > div:nth-child(5) > main > div:nth-child(5) > div:nth-child(4) > table > tbody > tr:nth-child('+str(i)+') > td.css-1u7zfla.e126mwsw1 > a').text
            show_title.append(show_name)
        except:
            break
    
    try:
        driver.find_element_by_css_selector('#app_mountpoint > div:nth-child(5) > main > div:nth-child(5) > a > div > button').click()
    except:
        pass
    
    time.sleep(2)
    
file = 'clean_data/netflix_data/all_netflix_titles.pkl'

pd.Series(show_title).to_pickle(directory + file)


