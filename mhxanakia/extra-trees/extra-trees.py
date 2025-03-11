# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:03:12 2025

@author: yiann
"""

#True to train - else it imports a model
train_model = True  

#True to optimize - else uses fixed hyperparameters
optimize_model = False  

model_imported = 'bestMhxanaki_ExtraTreesRegressor_RandomCV6.pkl'

#True for mapping - else no mapping
visualize = True  


#%% importations
print('Importing Libraries...')

import os
import numpy as np
import pandas as pd
import xarray as xr

os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras  #needed for NNs

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


print('Importing Data...')
dfLD = pd.read_parquet('df.parquet')
dfHD = pd.read_parquet('dfHD.parquet')

dfLD = dfLD[dfLD.valid_year<2020]
dfHD = dfHD[dfHD.valid_year>=2020]

#dfLD = dfLD.drop(columns=["valid_year"])
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# model gets trained
if train_model == True:
        
    best_model = ExtraTreesRegressor(n_estimators=85, 
                                     random_state=42, 
                                     max_depth=18,
                                     min_samples_leaf=4,
                                     min_samples_split=5,
                                     max_features=1.0
                                     )
    print(f'\nModel Used: {type(best_model).__name__}')
    print('Training with Fixed Hyperparameters...')
    best_model.fit(X_train, y_train)
    
    
'''
# model gets imported
else:
    print('Predicting Using Imported Model...')
    best_model = joblib.load(model_imported)
'''
    
    
# GIA CV THELEI OLO TO SET - GIA KANONIKA TO TEST - DIORTHWSE TO
# make predictions on hold-out set and show some stats
y_pred = best_model.predict(X_test)
random_mse = mean_squared_error(y_test, y_pred)
print(f"Hold-out Test set MSE of the Model: {random_mse:.4f}")

print('\nParameters of the Model:')
print(best_model.get_params())

features_in_model = best_model.feature_names_in_
feature_importances = best_model.feature_importances_
print("\nFeatures of the Model:", features_in_model)
print('Feature Importance:')
for i,feature in enumerate(features_in_model):
    print(f'{feature}:  {np.round(100*feature_importances[i],3)} %')

covariates = "-".join(features_in_model)
model_name = f'{type(best_model).__name__}_coraviates-{covariates}.pkl'
#joblib.dump(best_model, model_name)


#%% fitting the imported/optimized model t o high resolution data
print('Fitting on High Resolution Covariates...')
#better do column selection based on train_cols??

if 't2mHD' in dfHD.columns:
    downscale = dfHD.drop(['t2m', 't2mHD'], axis=1)
                                                 
else:
    downscale = dfHD.drop(['t2m'], axis=1)
    
downscale = best_model.predict(downscale)
dfHD['t2mHD'] = downscale
    
#to HD dataset paizei na vgazei komple pragmata kai xwris nan replacement
#isws xreiastei nan replacement gia thalasia merh mono
dfHD['t2mHD'] = dfHD['t2mHD'].where(dfHD['t2m'].notna()) #vazw NaN values
    
dsHD = xr.Dataset.from_dataframe(dfHD)
  
print("Done.")


#%% optikopoihsh
degree_spacing = 0.1
temporal_idx = 11
if visualize == True:
    t2mHD = dsHD.t2mHD - 273.15
    print('Mapping...')

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.LAKES.with_scale("50m"), 
                   edgecolor="black")
    
    #try the following with plt.pcolormesh() too
    #it just needs longitude, latitude, 2d-array
    #xr.plot assumes the center of the grid cell
    #plt.imshow assumes the top-left corner
    t2mHD.isel(valid_time=temporal_idx).plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cmap=plt.cm.inferno, 
        cbar_kwargs={"label": "Temperature (°C)"},
        #vmin=-1, vmax=360
    )
    
    ax.coastlines(resolution="10m", linewidth=0.75)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND)
    
    gl = ax.gridlines(draw_labels=True, linestyle=":", 
                      linewidth=0.5, color="k",
                      xlocs=np.arange(
                          dfLD.longitude.min(), 
                          dfLD.longitude.max(), 
                          degree_spacing
                          ),  #or: mticker.FixedLocator
                      ylocs=np.arange(
                          dfLD.latitude.max(), 
                          dfLD.latitude.min(), 
                          -degree_spacing
                          ) 
                      )
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlabel_values = ds.longitude.values[::5]
    #gl.ylabel_values = ds.latitude.values[::5]
    '''
    gl.xformatter = mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x in ds.longitude.values[::5] else ""
        )
    gl.yformatter = mticker.FuncFormatter(
        lambda y, _: f"{y:.1f}" if y in ds.latitude.values[::5] else ""
        )
    '''
    ax.set_title(
    (f"Downscaled Temperature "
     f"{np.datetime_as_string(dsHD.valid_time.values[temporal_idx], unit='M')}"
     )
        )
    #ax.set_title('Tem')
    plt.savefig(f'images-maps\\t2mHD-era5-land-{model_name}.png', dpi=1000)
    plt.show()
    
    #t2m = None
#%% sxolia
'''
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
    #mallon tha kanw bayesian optimization sto mellon...
    #gia kfold=5/n_iter=3 thelei 180sec konta...
    #an einai analogo, thelw polla parapanw gia exhaustive search...
    #mporei na dokimasw bayesian optimization kapoia stigmh...
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


