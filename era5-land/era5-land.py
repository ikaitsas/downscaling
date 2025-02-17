# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:42:11 2025

@author: yiann
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

extent = [39, 21, 36, 24]
years = list(range(1992,2023))
timescale = 'monthly'


#%% extracting file location - NEED TO MAKE THIS A FUNCTION
extent_string =  f'N{extent[0]}-W{extent[1]}-S{extent[-2]}-E{extent[-1]}'

subfolder = (
    f'extent__{extent_string}'
    )
folder = os.path.join("outputs", subfolder)
os.makedirs(folder, exist_ok=True)

filename = (
    f't2m-era5-land-{timescale}__'
    f'{extent_string}__'
    f'Period{years[0]}-{years[-1]}'
    f'.nc'
    )
file_path = os.path.join(folder, filename)


#%% xarray
ds = xr.open_dataset(file_path)

print(ds.variables)
print('\n')
print(ds.coords)
print('\n')
print('Variables: {}'.format(list(ds.keys())))
print('\n')
print('Coordinates: {}'.format(list(ds.coords)))


ds = ds.drop_isel(latitude=-1, longitude=-1)
t2m = ds.t2m.to_numpy()

df = ds.to_dataframe()
df['valid_time'] = df.index.get_level_values(0)
df['valid_month'] = df.valid_time.dt.month
df['valid_year'] = df.valid_time.dt.year
df['latitude'] = df.index.get_level_values(1)
df['longitude'] = df.index.get_level_values(2)
df = df.reset_index(drop=True)
df = df.drop(columns=['number', 'expver', 'valid_time'])


#%% create downscalign array
times_repeat = 10

t2mHD = np.repeat(
    np.repeat(t2m, times_repeat, axis=1), times_repeat, axis=2)

'''
# outdated version - works though
t2mHD = np.zeros(shape=(ds.valid_time.size, 
                        times_repeat*(ds.latitude.size), 
                        times_repeat*(ds.longitude.size))
                 )
# fill the HD version
for i in range(ds.valid_time.size):
    t2mHD_ = 0
    t2mHD_ = np.repeat(t2m[i,:,:], times_repeat, axis=0)
    t2mHD_ = np.repeat(t2mHD_, times_repeat, axis=1)
    t2mHD[i,:,:] = t2mHD_
'''

# apothikeush sto pc gia fortwma se montelaki
#np.save('t2m0.1deg.npy', t2m)
#np.save('t2m0.01deg.npy', t2mHD)


#%% map loading
degree_spacing = 0.1

print('\n')
print('Loading Map...')
#cyl is the Plate Caree projection - same as SRTM DEM data
m = Basemap(projection='cyl', 
            llcrnrlon=extent[1], 
            llcrnrlat=extent[-2], 
            urcrnrlon=extent[-1], 
            urcrnrlat=extent[0], 
            resolution='h')


#%% plotakia gia testakia
fig, ax = plt.subplots()

meridians = np.arange(extent[1], extent[-1] , degree_spacing)
parallels = np.arange(extent[-2], extent[0] , degree_spacing)

m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)

m.drawmeridians(meridians, linewidth=0.5)
m.drawmeridians(meridians[::5], linewidth=0.7, linestyle='-')

m.drawparallels(parallels, linewidth=0.5)
m.drawparallels(parallels[::5], linewidth=0.7, linestyle='-')
'''
cf = ax.contourf(ds.longitude, ds.latitude, t2m[11,:,:]-273.15,
                 cmap=plt.cm.inferno, levels=24, 
                 #vmin=np.nanmin(t2m-273.15),
                 #vmax=np.nanmax(t2m-273.15)
                 )
'''
valid_time_index = 11
cf = m.imshow(t2m[valid_time_index,:,:]-273.15, 
              cmap=plt.cm.inferno,
              origin='upper',
              aspect='equal'
              )  #cmap=plt.cm.PuOr_r
cbar = plt.colorbar(cf, pad=0, aspect=50)

cbar.set_label(f'Temperature [C]')
ax.set_title(f'T2m - {np.datetime_as_string(ds.valid_time.values[valid_time_index], unit="M")}')
plt.xticks(np.arange( extent[1], extent[-1] , 5*degree_spacing), 
           rotation=0, fontsize=8)
plt.yticks(np.arange( extent[-2], extent[0] , 5*degree_spacing), 
           rotation=0, fontsize=8)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#plt.savefig('t2m_era5land_resolution.png', dpi=300)
plt.show()

