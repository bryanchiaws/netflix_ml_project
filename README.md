# netflix_ml_project
Machine Learning Project - using neural networks to predict shows that I will like based on past viewing activity.

01 - Reading in Netflix viewing activity, downloaded from Netflix.com, concatenating data from different ccounts

02 - Scraping show genre, tags, number of episodes, episode length, etc. for each show from IMDb.com

03 - Cleaning the data

04 - Joining the data and getting it into a machine learning-ready format. Dropped features with high correlation to other features to reduce dimensionality

05 - Creating model, using cross validation to test model

06 - Parameter selection using grid search, multiprocessing to speed up grid search

07 - Model evaluation (looking at loss and accuracy)

08 - Scrape data for all available Netflix shows from reelgood.com, get show characteristics from IMDb.com

09 - Process and clean Netflix data

10 - Using X values generated in 09, predict probabilities of finishing each Netflix shows based on parameters derived in 07

Results available in 'final_predictions.xlsx'.
