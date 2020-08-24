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
import xlwt
import string
import numpy as np
import pandas as pd
from datetime import datetime as dt

directory = '/Users/bryanchia/Desktop/projects/netflix_ml_project/'

def get_tags(n):
    
    if (n%2 == 0):
        tag = driver.find_element_by_xpath('//*[@id="keywords_content"]/table/tbody/tr['+str(n/2+1)+']/td[1]/div[1]/a').text
    else:
        tag = driver.find_element_by_xpath('//*[@id="keywords_content"]/table/tbody/tr['+str((n+1)/2)+']/td[2]/div[1]/a').text
    
    return tag


def scrape(show_name):
    
    print(show_name)
       
    driver.implicitly_wait(0.5)

    driver.get('https://www.imdb.com/')
    
    
    driver.find_element_by_css_selector('#suggestion-search').send_keys(show_name)
        
    time.sleep(2)
    
    try:
        driver.find_element_by_xpath('//*[@id="react-autowhatever-1--item-0"]/a/div[2]/div[1]').click()
        show_title.append(show_name)
    except:
        show_title.append(show_name + 'NOT_FOUND')
        pass
    
    time.sleep(2)
    
    try:
        num_tags= int(re.sub('[\(\)]', '',(driver.find_element_by_xpath('//*[@id="titleStoryLine"]/div[2]/nobr/a').text).split()[2]))
        driver.find_element_by_xpath('//*[@id="titleStoryLine"]/div[2]/nobr/a').click()
        show_tag = []
        show_tag = [get_tags(n) for n in range(0, num_tags)]
        show_tags.append(show_tag)
       
    except:
        show_tags.append('')
        pass
        
    
    time.sleep(2)    
    
    driver.back()
    
    time.sleep(1)

n = 0

file = 'clean_data/netflix_data/all_netflix_titles.pkl'

ns = pd.read_pickle(directory + file)

unique_shows = list(ns)

#Call chromedriver
chromedriver = directory + '/chromedriver'
os.environ['webdriver.chrome.driver'] = chromedriver
driver = webdriver.Chrome(chromedriver)

show_title = []
show_tags = []
    
n = 0


for i in unique_shows:
    
    try:
        n +=1
        print(n)
        scrape(i)
    except:
        break
    
df_show_tags = pd.DataFrame(list(zip(show_title, show_tags)), 
               columns =['Title', 'Tags']) 

df_show_tags.to_pickle(directory + 'clean_data/netflix_data/show_tags_full.pkl')


