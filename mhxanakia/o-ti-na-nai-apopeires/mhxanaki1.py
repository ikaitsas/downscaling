# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:07:37 2025

@author: yiann
"""

import os
import numpy as np
import pandas as pd

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras


#%% ligo souloupwma akoma to dataset...
timeline = np.load('t2m_timeline.npy')

demLD = np.load('dem0.1deg.npy')
demLD1 = np.ravel(demLD, order='C') # like in df
demLD1 = np.tile(demLD1, timeline.size)


df = pd.read_parquet('t2m0.1deg.parquet')
df['DEM'] = demLD1      # we did it!!
df['month'] = df.valid_time.dt.month
df['year'] = df.valid_time.dt.year
#df.to_parquet('prwtoDatasetGiaMhxanaki.parquet')

df = df.drop(columns=['valid_time'])

#%% arxizei to training - prosexws!!
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

import matplotlib.pyplot as plt

dataset = df.dropna(subset=['t2m'])
#dataset.t2m = dataset.t2m 

# apo edw kai katw to xaos
train_cols = list(dataset.columns)[1:-1] #exclude t2m, year, #-2 for month
X = dataset.loc[:,train_cols] #mhpws thelei to t2m edw??
y = dataset.loc[:,'t2m']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=42, 
                              max_features=3
                              ) #n_jobs=-1?
model.fit(X_train, y_train)

#joblib.dump(model, 'prwto_mhxanaki_xwris_mhna.pkl')

feature_importances = model.feature_importances_
print("Feature importances:", feature_importances)

predictions = model.predict(X_test)
print("Predictions:", predictions)



plt.scatter(y_test-273.15, predictions-273.15, s=2)
plt.ylim([-5, 35])
plt.xlim([-5, 35])
plt.plot([-5, 35], [-5, 35], c='k', alpha=0.25)
plt.grid()
plt.xlabel('True values (y_test)')
plt.ylabel('Predicted Values')
plt.show()




import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MedAE: ', round(median_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

#print(regression_results(y_test, predictions))
'''  
    from statsmodels.api import OLS
    OLS(y_test,predictions2).fit().summary()
'''


#%% dokimes sto downscaled 2024
df2 = pd.read_parquet('t2m0.1deg2024.parquet')

timeline2024 = np.load('t2m_timeline2024.npy')
demLD2024 = np.load('dem0.1deg.npy')
demLD12024 = np.ravel(demLD2024, order='C') # like in df
demLD12024 = np.tile(demLD12024, timeline2024.size)

df2['DEM'] = demLD12024     # we did it!!
df2['month'] = df2.valid_time.dt.month
df2['year'] = df2.valid_time.dt.year

df2 = df2.drop(columns=['valid_time'])

#poso xwrizei thn kathe elaxisth ypodiairesh lat/lon
times_repeat = 10
df22 = df2.loc[df2.index.repeat(times_repeat*times_repeat)] 
#100 gia 10x10 fores kalyterh analysh
# menei na allaksw ta lat kai lon sto df22
# kai na treksw to montelo sto high resolution auto dataframe
# den nomizw na menei kati allo
# tha dw telika mhpws to montelo thelei kai to t2m sto training 
# wste na kanei to kalytero fit panw ston eauto tou??

# kane to na pairnei thn elaxisth ypodiairesh apo ta data
# kai nea elaxisth ypodiairesh: elaxisth/times_repeat
coco_lon = np.tile(np.arange(0,0.1,0.01), times_repeat)
coco_lon = np.tile(coco_lon,len(df2))
df22['longitude'] = df22.longitude+coco_lon #eurhka!

coco_lat = np.repeat(np.arange(0,-0.1,-0.01), times_repeat)
coco_lat = np.tile(coco_lat,len(df2))
df22['latitude'] = df22.latitude+coco_lat #yes!!

demHD = np.load('dem0.01deg.npy')
demHD1 = demHD.flatten(order='C')

#df22['DEMHD'] = np.tile(demHD1, timeline2024.size)
# VALE TO HIGH RESOLUTION DEM STO DF22!!!!

downscale = model.predict(df22.drop(['t2m', 'year'], axis=1))

#%%
'''
lat_start, lat_end, lat_step = 39.0, 36.0, -0.01
lon_start, lon_end, lon_step = 21.0, 24.0, 0.01

# Generate the latitude and longitude arrays
latitudes = np.arange(lat_start, lat_end + lat_step, lat_step)[:-1]
longitudes = np.arange(lon_start, lon_end + lon_step, lon_step)[:-2]

# Create a meshgrid
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

lat_lon_array = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
lat_array = np.row_stack

import xarray as xr
ds2 = xr.open_dataset('2024.nc')

t2m2 = ds2.t2m.to_numpy()
times_repeat = 10
t2mHD2 = np.zeros(shape=(ds2.valid_time.size, 
                         times_repeat*(ds2.latitude.size-1), 
                         times_repeat*(ds2.longitude.size-1))
                  )
for i in range(ds2.valid_time.size):
    t2mHD_2 = 0
    t2mHD_2 = np.repeat(t2m2[i,:-1,:-1], times_repeat, axis=0)
    t2mHD_2 = np.repeat(t2mHD_2, times_repeat, axis=1)
    t2mHD2[i,:,:] = t2mHD_2
'''




#%%
'''
t2mOG = np.load('t2m0.1deg.npy')
t2mHD = np.load('t2m0.01deg.npy')
demLD = np.load('dem0.1deg.npy')
demHD = np.load('dem0.01deg.npy')

#demLD1 = np.stack([demLD]*timeline.size, axis=2)
demLD1 = np.ravel(demLD, order='F')
#demLD1_ = np.tile(demLD1, timeline.size)
'''



