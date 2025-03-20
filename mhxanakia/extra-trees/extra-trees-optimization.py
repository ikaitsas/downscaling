# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:03:12 2025

@author: yiann
"""

#True to train - else it imports a model
train_model = True  

#True to optimize - else uses fixed hyperparameters
optimize_model = True  

model_imported = 'bestMhxanaki_ExtraTreesRegressor_RandomCV6.pkl'

#True for mapping - else no mapping
visualize = True  


#%% importations
print('Importing Libraries...')

import os
import time
import numpy as np
import pandas as pd
import xarray as xr

os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras  #needed for NNs

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


print('Importing Data...')
dfLD = pd.read_parquet('df.parquet')
#dfHD = pd.read_parquet('dfHD.parquet')

dfLD = dfLD[dfLD.valid_year<2020]
#dfHD = dfHD[dfHD.valid_year>=2020]

dfLD = dfLD.drop(columns=["valid_year"])
#dfHD = dfHD.drop(columns=["valid_year"])

#dfLD = dfLD.drop(columns=["aspect"])
#dfHD = dfHD.drop(columns=["aspect"])

#dfLD = dfLD.drop(columns=["slope"])
#dfHD = dfHD.drop(columns=["slope"])


#%% train/test split, import or train/optimize model
dfLD_ = dfLD.dropna(subset=['t2m'])
train_cols = list(dfLD_.columns)[1:] #exclude t2m
X = dfLD_.loc[:,train_cols] 
y = dfLD_.loc[:,'t2m']

# might be better to perform a train-test split non-randomly
# in order to capitalize the temporal consistency of the data?
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


#%% opt
# model gets trained
if optimize_model == True:
        
    model = ExtraTreesRegressor(random_state=42)
    print(f'\nModel Used: {type(model).__name__}')
    print('Starting Hyperparameter Optimization...')
    time_start = time.time()
    
    search_space = {
        "n_estimators": Integer(10, 250),  
        "max_depth": Categorical([None] + list(range(2, 25, 4))),  
        "min_samples_split": Integer(2, 20),  
        "min_samples_leaf": Integer(1, 20),  
        "max_features": Categorical(
            ['sqrt', 'log2', None] + list(
                np.round(np.arange(0.1,1.21,0.3),2)
                )
            ),  
        "bootstrap": Categorical([True, False])  
    }
    
    
    # might add a custom CV folding, for better representation
    # of time series data consistency?
    # for this reason, shuffling should be avoided...
    kfold = KFold(n_splits=10, shuffle=False, random_state=None)
    
    print("Performing BayesSearchCV...")
    bayes_search = BayesSearchCV(
        model,
        search_space,
        n_iter=100,  
        cv=kfold,  
        scoring="neg_mean_squared_error",
        n_jobs=1,  # -1 for all CPU cores
        verbose=3,
        random_state=42
    )
    # verbose only works for n_jobs=1 (others might work on linux?)
    # will probably add a logging setup to track progress
    
    bayes_search.fit(X_train, y_train)
    
    time_end = time.time()
    print(f"Hyperparameter Optimization Runtime: {time_end-time_start:.2f} s")
    
    print("\nBest parameters found by BayesSearchCV:")
    print(bayes_search.best_params_)
    print("\nBest cross-validated score (negative MSE):")
    print(bayes_search.best_score_)
    '''
    print('\nTraining Best Model on the Whole Dataset...')
    
    best_model = bayes_search.best_estimator_
    best_model.fit(X, y)
    '''

#%% sxolia
'''
# GIA CV THELEI TA TRAIN-VAL SETS - GIA KANONIKA TO TEST

# model is additionally optimized
if optimize_model == True:
    model = ExtraTreesRegressor(random_state=42)
    print(f'\nModel Used: {type(model).__name__}')
    print('Starting Hyperparameter Optimization...')
    time_start = time.time()
    
    param_grid = {
        'n_estimators': [25, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 10, 20],
        'max_features': ['sqrt', 'log2', None, 0.3]
    }
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Performing RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid, #param_distributions for RandomCV
        n_iter=50,  # Number of parameter settings sampled
        cv=kfold,
        scoring='neg_mean_squared_error',  #might also try MAE
        verbose=3,
        random_state=42, #enable for RandomCV
        n_jobs=1
    )
    #verbose only works for n_jobs=1 (others might work on linux?)
    
    random_search.fit(X_train, y_train)
    
    time_end = time.time()
    print(f"Hyperparameter Optimization Runtime: {time_end-time_start:.2f} s")
    
    print("\nBest parameters found by RandomizedSearchCV:")
    print(random_search.best_params_)
    print("\nBest cross-validated score (negative MSE):")
    print(random_search.best_score_)
    
    print('\nTraining Best Model on the Whole Dataset...')
    
    best_model = random_search.best_estimator_
    best_model.fit(X, y)
'''


