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


def scrape(show_name):
    
    print(show_name)
       
    driver.implicitly_wait(0.5)

    driver.get('https://www.imdb.com/')
    
    
    driver.find_element_by_css_selector('#suggestion-search').send_keys(show_name)
        
    time.sleep(3)
    
    try:
        driver.find_element_by_xpath('//*[@id="react-autowhatever-1--item-0"]/a/div[2]/div[1]').click()
        show_title.append(show_name)
    except:
        show_title.append(show_name + 'NOT_FOUND')
        pass
    
    time.sleep(3)

    try:
        show_length.append(driver.find_element_by_css_selector('#title-overview-widget > div.vital > div.title_block > div > div.titleBar > div.title_wrapper > div > time').text)
    except:
        show_length.append('')
        pass
    
    genres = []

    for i in range(0,4):
        try:
            genres.append(driver.find_element_by_xpath('//*[@id="title-overview-widget"]/div[1]/div[2]/div/div[2]/div[2]/div/a['+str(i+1)+']').text)
        except:
            break
       
    show_genre.append(genres)
    
    try:
        show_rating.append(driver.find_element_by_css_selector('#title-overview-widget > div.vital > div.title_block > div > div.ratings_wrapper > div.imdbRating > div.ratingValue > strong > span').text)
    except:
        show_rating.append('')
        pass
    
    try:
        show_episodes.append(driver.find_element_by_xpath('//*[@id="title-overview-widget"]/div[1]/div[3]/a/div/div/span').text)
    except:
        show_episodes.append('')
        pass
    
    try:
        show_seasons.append(driver.find_element_by_xpath('//*[@id="title-episode-widget"]/div/div[3]/a[1]').text)
    except:
        show_seasons.append('1')
        pass
    
    try:
        show_creators.append(driver.find_element_by_xpath('//*[@id="title-overview-widget"]/div[2]/div[1]/div[2]').text)
    except:
        show_creators.append('')
        pass
        
    actors = []

    for i in range(0,20):
        try:
            actors.append(driver.find_element_by_css_selector('#titleCast > table > tbody > tr:nth-child('+str(2*(i+1))+') > td:nth-child(2) > a').text)
        except:
            break
       
    show_actors.append(actors)
    
    driver.back()
    
    time.sleep(1)

file = 'clean_data/show_data.pkl'

n = 0

bv = pd.read_pickle(directory + file)

unique_shows = pd.Series(bv['Show'].unique()).dropna()

unique_shows = list(unique_shows[unique_shows != ' '])

#Call chromedriver
chromedriver = directory + '/chromedriver'
os.environ['webdriver.chrome.driver'] = chromedriver
driver = webdriver.Chrome(chromedriver)


show_title = []
show_length = []
show_genre = []
show_rating = []
show_actors = []
show_episodes = []
show_seasons = []
show_creators = []
    
n = 0


for i in unique_shows:
    
    try:
        n +=1
        print(n)
        scrape(i)
    except:
        break
    
df_show_chars = pd.DataFrame(list(zip(show_title, show_rating, show_length, show_genre, show_episodes, show_seasons)), 
               columns =['Title', 'Rating', 'Length', 'Genre', 'Episodes', 'Total Seasons']) 

df_show_chars.to_pickle(directory + '/show_chars_full.pkl')

df_show_actors = pd.DataFrame(list(zip(show_title, show_actors)), 
               columns =['Title', 'Actors']) 

df_show_actors.to_pickle(directory + '/show_actors_full.pkl')

df_show_creators = pd.DataFrame(list(zip(show_title, show_creators)), 
               columns =['Title', 'Creators']) 

df_show_creators.to_pickle(directory + '/show_creators_full.pkl')

