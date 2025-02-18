# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:42:11 2025

@author: yiann

Tidying up ERA5-Land data and import DEM-derived morphography metadata 
in the respective coordinates

Will probably add land cover data in here too
And any other data needes, either as keys, coordinates, dimensions,
straight to the Datasets and DataArrays...
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

extent = [39, 21, 36, 24]
years = list(range(1992,2023))
timescale = 'monthly'
visualize = False

# make those be imported via their path to the DEM folder
dem = np.load("dem_coco0.1deg_0.1deg.npy")
demHD = np.load("dem_coco0.01deg_0.01deg.npy")


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


#%% xarray - pandas
ds = xr.open_dataset(file_path)

ds = ds.drop_isel(latitude=-1, longitude=-1)
#parakatw kanei slice, oxi removal, alla epeidh kanw
#reassign sthn idia metablhth, praktika prokyptei to idio
#ds = ds.isel(latitude=slice(None, -1))

ds.coords["dem"] = (["latitude", "longitude"], dem)
ds.coords["dem"].attrs["units"] = "meters"
ds.coords["dem"].attrs["description"] = "Elevation at each lat-lon pair"

if "number" in ds.coords:
    ds = ds.reset_coords(["number"], drop=True)
if "expver" in ds.coords:
    ds = ds.reset_coords(["expver"], drop=True)


t2m_array = ds.t2m.to_numpy()

df = ds.to_dataframe()
if "number" in df.columns:
    df = df.drop(columns=['number'])
if "expver" in df.columns:
    df = df.drop(columns=['expver'])
    
t2mColumn = "t2m"  #list(ds.keys())[i] more generally - i=0 here
t2mColumn1st = [t2mColumn] + [col for col in df.columns if col!=t2mColumn]
df = df[t2mColumn1st]  #ensure t2m is always first column

if "valid_time" in df.index.names:
    df['valid_time'] = df.index.get_level_values("valid_time")  #lvl-0
    df['valid_year'] = df.valid_time.dt.year
    df['valid_month'] = df.valid_time.dt.month

if "latitude" in df.index.names:
    df['latitude'] = df.index.get_level_values("latitude")      #lvl-1
if "longitude" in df.index.names:
    df['longitude'] = df.index.get_level_values("longitude")    #lvl-2

if "valid_time" in df.columns:
    df = df.drop(columns=['valid_time'])
#df = df.reset_index(drop=True)

#will probably make the column order specified explicitly, to
#avoid any column order mismatches for ML & NN training, t2m
#order specification is not enough...
#will be done to dfHD below too...

print(ds.variables)
print('\n')
print(ds.coords)
print('\n')
print('Keys: {}'.format(list(ds.keys())))
print('\n')
print('Coordinates: {}'.format(list(ds.coords)))


#%% create downscalign HD array
era5Land_resolution = 0.1
scaling_factor = 10  #divide resolution by this number

#axis=1 latitude, axis=2 longitude, change accordingly
#will probably add index extraction from Dataset dimensions
t2mHD_array = np.repeat(
    np.repeat(t2m_array, scaling_factor, axis=1), scaling_factor, axis=2)

'''
# outdated repeat version - works though
t2mHD = np.zeros(shape=(ds.valid_time.size, 
                        scaling_factor*(ds.latitude.size), 
                        scaling_factor*(ds.longitude.size))
                 )
# fill the HD version
for i in range(ds.valid_time.size):
    t2mHD_ = 0
    t2mHD_ = np.repeat(t2m[i,:,:], scaling_factor, axis=0)
    t2mHD_ = np.repeat(t2mHD_, scaling_factor, axis=1)
    t2mHD[i,:,:] = t2mHD_
'''

scaled_coords = {
    #automatopoihse thn epilogh twn min/max sta lat/lon
    'valid_time': ds.t2m.valid_time.values,
    'latitude': np.arange(39,36,-era5Land_resolution/scaling_factor),
    'longitude': np.arange(21,24,era5Land_resolution/scaling_factor), 
}


t2mHD = xr.DataArray(t2mHD_array,
                     coords=scaled_coords,
                     dims=["valid_time", "latitude", "longitude"]
                     )

t2mHD.coords["dem"] = (["latitude", "longitude"], demHD)

t2mHD.coords["dem"].attrs["units"] = "meters"
t2mHD.coords["dem"].attrs["description"] = "Elevation at each lat-lon pair"


dfHD = t2mHD.to_dataframe(name='t2m')

t2mColumn = "t2m"  #list(ds.keys())[i] more generally - i=0 here
t2mColumn1st = [t2mColumn] + [col for col in dfHD.columns if col!=t2mColumn]
dfHD = dfHD[t2mColumn1st]  #ensure t2m is always first column

if "valid_time" in dfHD.index.names:
    dfHD['valid_time'] = dfHD.index.get_level_values("valid_time")  #lvl-0
    dfHD['valid_year'] = dfHD.valid_time.dt.year
    dfHD['valid_month'] = dfHD.valid_time.dt.month

if "latitude" in dfHD.index.names:
    dfHD['latitude'] = dfHD.index.get_level_values("latitude")      #lvl-1
if "longitude" in dfHD.index.names:
    dfHD['longitude'] = dfHD.index.get_level_values("longitude")    #lvl-2

if "valid_time" in dfHD.columns:
    dfHD = dfHD.drop(columns=['valid_time'])
#dfHD = dfHD.reset_index(drop=True)


#%% map loading
if visualize == True:
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
    cf = m.imshow(t2m_array[valid_time_index,:,:]-273.15, 
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

