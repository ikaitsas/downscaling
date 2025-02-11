# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:54:44 2025

@author: yiann
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


dsLD = xr.open_dataset('era5land.nc')
demLD = np.load('dem0.1deg.npy')

dsHD = xr.open_dataset('2024.nc')
demHD = np.load('dem0.01deg.npy')


#%% enswmatwsh DEM gia to LD - 1994me2023
dsLD = dsLD.isel(latitude=slice(None, -1))
dsLD = dsLD.isel(longitude=slice(None, -1))

dsLD['DEM'] = (['valid_time','latitude', 'longitude'], 
               np.tile(demLD, (dsLD.valid_time.size, 1, 1))
               )

dfLD = dsLD.to_dataframe()
dfLD['valid_time'] = dfLD.index.get_level_values(0)
dfLD['latitude'] = dfLD.index.get_level_values(1)
dfLD['longitude'] = dfLD.index.get_level_values(2)
#dfLD = dfLD.reset_index(drop=True)
dfLD['month'] = dfLD.valid_time.dt.month
dfLD['year'] = dfLD.valid_time.dt.year
dfLD = dfLD.drop(columns=['valid_time', 'number', 'expver', 'skt'])
#dfLD.to_parquet('dfLD.parquet')


#%% enswmatwsh DEM gia to HD - 2024
times_repeat = 10   #scale factor

dsHD = dsHD.isel(latitude=slice(None, -1))
dsHD = dsHD.isel(longitude=slice(None, -1))

t2m = dsHD.t2m
scaled_data = np.repeat(t2m.values, times_repeat, axis=1)
scaled_data = np.repeat(scaled_data, times_repeat, axis=2)

scaled_coords = {
    #automatopoihse thn epilogh twn min/max sta lat/lon
    'valid_time': t2m.valid_time.values,
    'latitude': np.arange(39,36,-0.01),
    'longitude': np.arange(21,24,0.01), 
}

t2mHD = xr.DataArray(
    scaled_data,
    coords=scaled_coords,
    dims=['valid_time', 'latitude', 'longitude']
)

dsHD = t2mHD.to_dataset(name='t2m')

dsHD['DEM'] = (['valid_time','latitude', 'longitude'], 
               np.tile(demHD, (dsHD.valid_time.size, 1, 1))
               )

dfHD = dsHD.to_dataframe()
dfHD['valid_time'] = dfHD.index.get_level_values(0)
dfHD['latitude'] = dfHD.index.get_level_values(1)
dfHD['longitude'] = dfHD.index.get_level_values(2)
#dfLD = dfLD.reset_index(drop=True)
dfHD['month'] = dfHD.valid_time.dt.month
dfHD['year'] = dfHD.valid_time.dt.year
dfHD = dfHD.drop(columns=['valid_time'])
#dfHD.to_parquet('dfHD.parquet')


#%% meo mhxanaki training
dfLD_ = dfLD.dropna(subset=['t2m'])

train_cols = list(dfLD_.columns)[1:-1] #exclude t2m, year, #-2 for month
X = dfLD_.loc[:,train_cols] #mhpws thelei to t2m edw??
y = dfLD_.loc[:,'t2m']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

model2 = RandomForestRegressor(n_estimators=50, random_state=42, 
                              max_features=2
                              ) #n_jobs=-1?
model2.fit(X_train, y_train)

#joblib.dump(model2, 'mhxanaki2.pkl')

fueatures_in_model = model2.feature_names_in_
print("Features of the model:", fueatures_in_model)
feature_importances = model2.feature_importances_
print("Feature importances:", feature_importances)

predictions2 = model2.predict(X_test)
#print("Predictions:", predictions2)


# ena tsapatsouliko  true/modeled plotaki
plt.scatter(y_test-273.15, predictions2-273.15, s=2)
plt.ylim([-5, 35])
plt.xlim([-5, 35])
plt.plot([-5, 35], [-5, 35], c='k', alpha=0.25)
plt.grid()
plt.xlabel('True values (y_test)')
plt.ylabel('Predicted Values')
plt.axis('scaled')
plt.show()


#%% dokimazontas efarmogh montelakiou sto HD DEM gia downscaling
if 't2mHD' in dfHD.columns:
    downscale = model2.predict(dfHD.drop(['t2m', 'year', 't2mHD'], axis=1))
else:
    downscale = model2.predict(dfHD.drop(['t2m', 'year'], axis=1))

dfHD['t2mHD'] = downscale

dfHD['t2mHD'] = dfHD['t2mHD'].where(dfHD['t2m'].notna()) #vazw NaN values

dsHD2 = xr.Dataset.from_dataframe(dfHD)


#%% map loading
degree_spacing = 0.1
#lat = np.flipud(np.unique( df.index.get_level_values(0).to_numpy() ))
# to lat thelei flip ste na exei thn seira tou datasheet, prepei to unique na
# to kanei sort logika... to lon einai hdh sorted, den exei thema...
#lon = np.unique( df.index.get_level_values(1).to_numpy() )
#cyl is the Plate Caree projection - same as SRTM DEM data
m = Basemap(projection='cyl', llcrnrlon=21, llcrnrlat=36, 
            urcrnrlon=24, urcrnrlat=39, resolution='h')


#%% test plotakia
fig, ax = plt.subplots()
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.drawmeridians(np.arange(21, 24 , degree_spacing),
                linewidth=0.5)
m.drawparallels(np.arange(36, 39 , degree_spacing),
                linewidth=0.5)
cf = m.imshow(np.flipud(dsHD2.t2m.values[11,:,:]-273.15), 
              cmap=plt.cm.inferno)
cbar = plt.colorbar(cf, pad=0, aspect=50)
cbar.set_label(f'Temperature [C]')
ax.set_title('T2m - 0.1deg Original')
plt.xticks(np.arange( 21,24+5*degree_spacing , 5*degree_spacing))
plt.yticks(np.arange( 36, 39+5*degree_spacing , 5*degree_spacing))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('0_KATI_KANAME1.png', dpi=300)
plt.show()


fig, ax = plt.subplots()
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.drawmeridians(np.arange(21, 24 , degree_spacing),
                linewidth=0.5)
m.drawparallels(np.arange(36, 39 , degree_spacing),
                linewidth=0.5)
cf = m.imshow(np.flipud(dsHD2.t2mHD.values[11,:,:]-273.15), 
              cmap=plt.cm.inferno)
cbar = plt.colorbar(cf, pad=0, aspect=50)
cbar.set_label(f'Temperature [C]')
ax.set_title('T2m  - 0.01deg Downscaled')
plt.xticks(np.arange( 21,24+5*degree_spacing , 5*degree_spacing))
plt.yticks(np.arange( 36, 39+5*degree_spacing , 5*degree_spacing))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('0_KATI_KANAME2.png', dpi=300)
plt.show()

'''
plt.imshow(dsHD.t2m.values[11,:,:]-273.15, cmap=plt.cm.inferno)
plt.show()
'''

