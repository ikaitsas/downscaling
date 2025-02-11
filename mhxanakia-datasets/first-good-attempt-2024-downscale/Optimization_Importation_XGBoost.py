# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:03:12 2025

@author: yiann
"""

#True to train - else it imports a model
train_model = True 

#True to optimize - else uses fixed hyperparameters
optimize_model = False  

model_imported = 'bestMhxanaki_XGBRegressor_RandomCV.pkl'

#True for mapping - else no mapping
perform_mapping_operations = False  


#%% importations
print('Importing Libraries...')

import os
import time
import numpy as np
import pandas as pd
import xarray as xr

os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras  #needed for NNs

import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


print('Importing Data...')

dsLD = xr.open_dataset('dsLD.nc')
dsHD = xr.open_dataset('dsHD.nc')

dfLD = pd.read_parquet('dfLD.parquet')
dfHD = pd.read_parquet('dfHD.parquet')
dfHD = dfHD.drop(columns=['t2mHD'])


#%% train/test split, import or train/optimize model
dfLD_ = dfLD.dropna(subset=['t2m'])
train_cols = list(dfLD_.columns)[1:-1] #exclude t2m, year, #-2 for month
X = dfLD_.loc[:,train_cols]
X = dfLD_.loc[:,train_cols] 
y = dfLD_.loc[:,'t2m']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# model gets trained
if train_model == True:
    
    # model is additionally optimized
    if optimize_model == True:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
        print(f'\nModel Used: {type(model).__name__}')
        print('Starting Hyperparameter Optimization...')
        time_start = time.time()
        
        #https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
        param_grid = {
            'n_estimators': [250, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
            'max_depth': [2, 6, 10, 20],
            'min_child_weight': [1, 5, 10, 50],
            'subsample': [0.1, 0.5, 0.8],
            'colsample_bytree': [0.1, 0.5, 0.8],
            #'gamma': [0, 0.25, 0.5, 1, 10],
            #'reg_lambda: [1, 5, 10, 100]
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
        #mallon tha kanw bayesian optimization sto mellon...
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
    
    else:
        
        best_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
        
        best_model.set_params(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=40,
            subsample=0.5,
            colsample_bytree=0.5,
            gamma=25,
            reg_lambda=1,
            reg_alpha = 1
            )
        
        print(f'\nModel Used: {type(best_model).__name__}')
        print('Training with Fixed Hyperparameters...')
        best_model.fit(X_train, y_train)
    '''
    joblib.dump(best_model, 
                f'bestMhxanaki_{type(best_model).__name__}_RandomCV12.pkl')
    '''
# model gets imported
else:
    print('Predicting Using Imported Model...')
    best_model = joblib.load(model_imported)
    
    
# GIA CV THELEI OLO TO SET - GIA KANONIKA TO TEST - DIORTHWSE TO
# make predictions on hold-out set and show some stats
y_pred = best_model.predict(X_test)

train_mse = mean_squared_error(y_train, best_model.predict(X_train))
print(f"Train set MSE of the Model: {train_mse:.4f}")

test_mse = mean_squared_error(y_test, y_pred)
print(f"Hold-out Test set MSE of the Model: {test_mse:.4f}")

print('\nParameters of the Model:')
print(best_model.get_params())

fueatures_in_model = best_model.feature_names_in_
feature_importances = best_model.feature_importances_
print("\nFeatures of the Model:", fueatures_in_model)
print('Feature Importance:')
for i,fueature in enumerate(fueatures_in_model):
    print(f'{fueature}:  {np.round(100*feature_importances[i],3)} %')


#%% fitting the imported/optimized model t o high resolution data
print('Fitting on High Resolution Covariates...')
#better do column selection based on train_cols??

if 't2mHD' in dfHD.columns:
    downscale = dfHD.drop(['t2m', 'year', 't2mHD'], axis=1)
                                                 
else:
    downscale = dfHD.drop(['t2m', 'year'], axis=1)
    
downscale = best_model.predict(downscale)
dfHD['t2mHD'] = downscale
    
#to HD dataset paizei na vgazei komple pragmata kai xwris nan replacement
#isws xreiastei nan replacement gia thalasia merh mono
dfHD['t2mHD'] = dfHD['t2mHD'].where(dfHD['t2m'].notna()) #vazw NaN values
    
dsHD2 = xr.Dataset.from_dataframe(dfHD)
  
print("Done.")


#%% map loading
if perform_mapping_operations == True:
    print('\nLoading Map...')
    degree_spacing = 0.1
    #cyl is the Plate Caree projection - same as SRTM DEM data
    m = Basemap(projection='cyl', llcrnrlon=21, llcrnrlat=36, 
                urcrnrlon=24, urcrnrlat=39, resolution='h') 
    #automatopoihsh later


#%% test plotakia
    print('Plotting Coarse Resolution...')
    fig, ax = plt.subplots()
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawmeridians(np.arange(21, 24 , degree_spacing),
                    linewidth=0.5)
    m.drawparallels(np.arange(36, 39 , degree_spacing),
                    linewidth=0.5)
    cf = m.imshow(dsHD2.t2m.values[11,:,:]-273.15, 
                  cmap=plt.cm.inferno)
    cbar = plt.colorbar(cf, pad=0, aspect=50)
    cbar.set_label(f'Temperature [C]')
    ax.set_title('T2m - 0.1deg Original')
    plt.xticks(np.arange( 21,24+5*degree_spacing , 5*degree_spacing ))
    plt.yticks(np.arange( 36, 39+5*degree_spacing , 5*degree_spacing ))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.savefig('0_KATI_KANAME22OG.png', dpi=300)
    plt.show()
    
    print('Plotting High Resolution...')
    fig, ax = plt.subplots()
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawmeridians(np.arange(21, 24 , degree_spacing),
                    linewidth=0.25)
    m.drawparallels(np.arange(36, 39 , degree_spacing),
                    linewidth=0.25)
    cf = m.imshow(dsHD2.t2mHD.values[11,:,:]-273.15, 
                  cmap=plt.cm.inferno)
    cbar = plt.colorbar(cf, pad=0, aspect=50)
    cbar.set_label(f'Temperature [C]')
    ax.set_title('T2m  - 0.01deg Downscaled')
    plt.xticks(np.arange( 21,24+5*degree_spacing , 5*degree_spacing ))
    plt.yticks(np.arange( 36, 39+5*degree_spacing , 5*degree_spacing ))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.savefig('0_KATI_KANAME22XGB12.png', dpi=300)
    plt.show()
    
    print('Done.')



