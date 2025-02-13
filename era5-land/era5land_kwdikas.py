# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:01:34 2024

@author: yiann
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

file_name = 'era5land.nc'
#file_name = 'temp2mERA5monthly_1994to2023.nc'
ds = xr.open_dataset(file_name)

print(ds.variables)
print('\n')
print(ds.coords)
print('\n')
print('Variables: {}'.format(list(ds.keys())))
print('\n')
print('Coordinates: {}'.format(list(ds.coords)))

t2m = ds.t2m.to_numpy()
lst = ds.skt.to_numpy()


df = ds.drop_isel(latitude=-1, longitude=-1).to_dataframe()
df['valid_time'] = df.index.get_level_values(0)
df['latitude'] = df.index.get_level_values(1)
df['longitude'] = df.index.get_level_values(2)
df = df.reset_index(drop=True)
df = df.drop(columns=['number', 'expver'])#, 'skt'])
df.to_parquet('t2m0.1deg.parquet')


'''
from scipy.signal import convolve2d
kernel = np.ones((3600,3600))
convolved = convolve2d(t2m[11,:,:], kernel, mode='valid')
'''
#t2mHD_size = np.array(t2m[11,:-1,:-1].shape)*360+1
#t2mHD_test = t2m[0,:-1,:-1].copy()

#epanalhpsh epi times_repeat - 
times_repeat = 10
t2mHD = np.zeros(shape=(ds.valid_time.size, 
                        times_repeat*(ds.latitude.size-1), 
                        times_repeat*(ds.longitude.size-1))
                 )
# fill the HD version
for i in range(ds.valid_time.size):
    t2mHD_ = 0
    t2mHD_ = np.repeat(t2m[i,:-1,:-1], times_repeat, axis=0)
    t2mHD_ = np.repeat(t2mHD_, times_repeat, axis=1)
    t2mHD[i,:,:] = t2mHD_

# apothikeush sto pc gia fortwma se montelaki
np.save('t2m0.1deg.npy', t2m[:,:-1,:-1])
np.save('t2m0.01deg.npy', t2mHD)
np.save('t2m_timeline.npy', ds.valid_time)


#%% map loading
degree_spacing = 0.5
df = ds.to_dataframe().droplevel('valid_time')
lat = np.flipud(np.unique( df.index.get_level_values(0).to_numpy() ))
# to lat thelei flip ste na exei thn seira tou datasheet, prepei to unique na
# to kanei sort logika... to lon einai hdh sorted, den exei thema...
lon = np.unique( df.index.get_level_values(1).to_numpy() )
#cyl is the Plate Caree projection - same as SRTM DEM data
m = Basemap(projection='cyl', llcrnrlon=np.min(lon), llcrnrlat=np.min(lat), 
            urcrnrlon=np.max(lon), urcrnrlat=np.max(lat), resolution='h')


#%% plotakia gia testakia
fig, ax = plt.subplots()

m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.drawmeridians(np.arange(np.min(lon), np.max(lon) , degree_spacing),
                linewidth=0.5)
m.drawparallels(np.arange(np.min(lat), np.max(lat) , degree_spacing),
                linewidth=0.5)
'''
cf = ax.contourf(ds.longitude, ds.latitude, t2m[11,:,:]-273.15,
                 cmap=plt.cm.PuOr_r, levels=24, 
                 #vmin=np.nanmin(t2m-273.15),
                 #vmax=np.nanmax(t2m-273.15)
                 )
'''
cf = m.imshow(np.flipud(t2m[11,:-1,:-1]-273.15), cmap=plt.cm.inferno)  #cmap=plt.cm.PuOr_r
#cf = m.imshow(np.flipud(t2mHD[:,:]-273.15), cmap=plt.cm.inferno)  #cmap=plt.cm.PuOr_r
cbar = plt.colorbar(cf, pad=0, aspect=50)
'''
plt.scatter(df.index.get_level_values('longitude'), 
            df.index.get_level_values('latitude'), 
            c='k', s=0.01, alpha=0.1)
'''
cbar.set_label(f'Temperature [C]')
#ax.barbs(ds.longitude.values, ds.latitude.values, ds.u.values, ds.v.values, 
#         color='black', length=5, alpha=0.5)
'''
lon_grid, lat_grid = np.meshgrid(ds.longitude.values, ds.latitude.values)
mporei na diorthwthei mallon to sfalma sthn barbs???
p.x. ax.barbs(lon_grid, lat_grid, ds.u.values, ds.v.values, color='black', 
              length=5, alpha=0.5)
alla edw exw kathe 0.25 moires velaki, opote polla shmeia kai den fainetai
tipota kala gia na dw an einai swsta..
'''
ax.set_title('T2m')
plt.xticks(np.arange( np.min(lon), np.max(lon) , degree_spacing))
plt.yticks(np.arange( np.min(lat), np.max(lat) , degree_spacing))
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.savefig('t2m_rea5land_resolution.png', dpi=300)
plt.show()