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

print('Importing libraris and data...')
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from mpl_toolkits.basemap import Basemap


extent = [39, 21, 36, 24]
years = list(range(1992,2023))
timescale = 'monthly'
visualize = False
save_dfs = False


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


#%% normal array from ERA5-Land
ds = xr.open_dataset(file_path)

ds = ds.drop_isel(latitude=-1, longitude=-1)
#parakatw kanei slice, oxi removal, alla epeidh kanw
#reassign sthn idia metablhth, praktika prokyptei to idio
#ds = ds.isel(latitude=slice(None, -1))

print('Merging with the ERA5-Land resolution DEM')
ds.coords["dem"] = (["latitude", "longitude"], dem)
ds.coords["dem"].attrs["units"] = "meters"
ds.coords["dem"].attrs["description"] = "Elevation at each lat-lon pair"

if "number" in ds.coords:
    ds = ds.reset_coords(["number"], drop=True)
if "expver" in ds.coords:
    ds = ds.reset_coords(["expver"], drop=True)


t2m_array = ds.t2m.to_numpy()

era5Land_resolution = 0.1
print(f'Extracting dataframe from {era5Land_resolution}deg ERA5-Land data...')
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
if save_dfs == True:
    df.to_parquet("df.parquet")

#will probably make the column order specified explicitly, to
#avoid any column order mismatches for ML & NN training, t2m
#order specification is not enough...
#will be done to dfHD below too...
'''
print(ds.variables)
print('\n')
print(ds.coords)
print('\n')
print('Keys: {}'.format(list(ds.keys())))
print('\n')
print('Coordinates: {}'.format(list(ds.coords)))
'''


#%% create downscalign HD array
scaling_factor = 10  #divide resolution by this number

print('Producing HD version...')
print(f'Dividing each grid cell {scaling_factor}x{scaling_factor} times...')
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


print(f'Extracting {era5Land_resolution/scaling_factor}deg HD dataframe...')
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
if save_dfs == True:
    dfHD.to_parquet("dfHD.parquet")


#%%
valid_time_index = 11
degree_spacing = 0.1


if visualize == True:
    print('Mapping...')

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    #try the following with plt.pcolormesh() too
    #xr.plot assumes the center of the grid cell
    #plt.imshow assumes the top-left corner
    ds.t2m.isel(valid_time=valid_time_index).plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cmap=plt.cm.inferno, 
        cbar_kwargs={"label": "Temperature (°C)"}
    )
    
    ax.coastlines(resolution="10m", linewidth=0.75)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES.with_scale("10m"))
    ax.add_feature(cfeature.LAND)
    
    gl = ax.gridlines(draw_labels=True, linestyle=":", 
                      linewidth=0.5, color="k",
                      xlocs=np.arange(
                          ds.longitude.values.min(), 
                          ds.longitude.values.max(), 
                          degree_spacing
                          ),  #or: mticker.FixedLocator
                      ylocs=np.arange(
                          ds.latitude.values.max(), 
                          ds.latitude.values.min(), 
                          -degree_spacing
                          ) 
                      )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_values = ds.longitude.values[::5]
    gl.ylabel_values = ds.latitude.values[::5]
    
    '''
    gl.xformatter = mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x in ds.longitude.values[::5] else ""
        )
    gl.yformatter = mticker.FuncFormatter(
        lambda y, _: f"{y:.1f}" if y in ds.latitude.values[::5] else ""
        )
    '''
    ax.set_title(f"Temperature {np.datetime_as_string(ds.valid_time.values[valid_time_index], unit='M')}")
    #plt.savefig('images-maps\\t2m-era5-land-cartopy.png', dpi=1000)
    plt.show()


print('Done')


'''
#------------------------------------------------------------------------------
#advanced cartopy plotting - based on eumetsat autumn course

# choose the plot size (width x height, in inches)
plt.figure(figsize=(10,10))

# use the PlateCarree projection in cartopy
ax = plt.axes(projection=ccrs.PlateCarree())

# define the image extent
img_extent = [lons.min(), lons.max(), lats.min(), lats.max()]


# plot the image
img = ax.imshow(data, vmin=-10, vmax=50, origin='upper', extent = img_extent, 
                cmap='jet',interpolation = 'None')


# add some various map elements to the plot
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

# add coastlines, borders and gridlines
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.4)
gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  color='gray', 
                  alpha=1.0, 
                  linestyle='--', 
                  linewidth=0.25,
                  xlocs=np.arange(-180, 181, 10), 
                  ylocs=np.arange(-90, 91, 10), 
                  draw_labels=True)

gl.top_labels = False
gl.right_labels = False

# add a colorbar
plt.colorbar(img, label='Land Surface Temperature - All Sky (°C)', 
             extend='both', orientation='horizontal',
             pad=0.05, fraction=0.04)

# get the date
date_str  = file.getncattr('time_coverage_start')
date_format = '%Y-%m-%dT%H:%M:%SZ'
date_obj = datetime.strptime(date_str, date_format)
date = date_obj.strftime('%Y-%m-%d %H:%M:%S UTC')

# add a title
plt.title(f'MSG/SEVIRI -  LST - All Sky \n{date}', fontweight='bold', 
          fontsize=10, loc='left')
plt.title('Autumn School 2024', fontsize=10, loc='right')

# save the image
plt.savefig('OUTPUT/image_LST-AS_2.png')

# show the image
plt.show()


#------------------------------------------------------------------------------
plotting reginal map with shapefile features - eumetsat autumn school

# choose the plot size (width x height, in inches)
plt.figure(figsize=(10,10))

# use the PlateCarree projection in cartopy
ax = plt.axes(projection=ccrs.PlateCarree())

#extent = [6,35, 20,48] #[lonmin,latmin, lonmax,latmax]
#extent = [-10,35, 5,45] #[lonmin,latmin, lonmax,latmax]
extent = [17,34, 32,44] #[lonmin,latmin, lonmax,latmax]

# latitude lower and upper index
latli = np.argmin( np.abs( lats - extent[1] ) )
latui = np.argmin( np.abs( lats - extent[3] ) )

# longitude lower and upper index
lonli = np.argmin( np.abs( lons - extent[0] ) )
lonui = np.argmin( np.abs( lons - extent[2] ) )

# extract the data
data = file.variables['MLST-AS'][ 0 , latui:latli , lonli:lonui ]

# define the image extent
img_extent = [extent[0], extent[2], extent[1], extent[3]]

# plot the image
img = ax.imshow(data, vmin=-10, vmax=50, origin='upper', extent=img_extent, 
                cmap='RdYlBu_r',interpolation='None')

# add coastlines, borders and gridlines
ax.coastlines(resolution='10m', color='lightgrey', linewidth=0.8)
ax.add_feature(cartopy.feature.BORDERS, edgecolor='darkgrey', linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, 
                  linestyle='--', linewidth=0.25, 
                  xlocs=np.arange(-180, 181, 5), ylocs=np.arange(-90, 91, 5), 
                  draw_labels=True)
gl.top_labels = False
gl.right_labels = False
gl.xpadding = -5
gl.ypadding = -5


# add some elements to the plot
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

# add a colorbar

# add a shapefile
shapefile = list(shpreader.Reader(
    'auxfiles/gadm41_ITA/gadm41_ITA_1.shp'
    ).geometries())
ax.add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='black',
                  facecolor='none', linewidth=0.3)
plt.colorbar(img, label='Land Surface Temperature -All Sky (°C)', 
             extend='both', orientation='horizontal', pad=0.045, 
             fraction=0.045)

# add a title
plt.title(f'MSG/SEVIRI -  LST - All Sky \n{date}', fontweight='bold', 
          fontsize=10, loc='left')
plt.title('Autumn School 2024', fontsize=10, loc='right')

#----------------------

# save the image
plt.savefig('OUTPUT/image_LST-AS_3.png')

# show the image
plt.show()


#------------------------------------------------------------------------------
#using basemap for plotting - not recommended - map loads slowly

print('Loading map...')
if visualize == True:
    
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
    
    #cf = ax.contourf(ds.longitude, ds.latitude, t2m[11,:,:]-273.15,
    #                 cmap=plt.cm.inferno, levels=24, 
    #                 #vmin=np.nanmin(t2m-273.15),
    #                 #vmax=np.nanmax(t2m-273.15)
    #                 )
    
    cf = m.imshow(t2m_array[valid_time_index,:,:]-273.15, 
                  cmap=plt.cm.coolwarm,
                  origin='upper',
                  aspect='equal'
                  )  #cmap=plt.cm.inferno
    cbar = plt.colorbar(cf, pad=0, aspect=50)
    
    cbar.set_label(f'Temperature [C]')
    ax.set_title(f'T2m - {np.datetime_as_string(ds.valid_time.values[valid_time_index], unit="M")}')
    plt.xticks(np.arange( extent[1], extent[-1] , 5*degree_spacing), 
               rotation=0, fontsize=8)
    plt.yticks(np.arange( extent[-2], extent[0] , 5*degree_spacing), 
               rotation=0, fontsize=8)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    #plt.savefig('images-maps\\t2metraa_era5land_resolution.png', dpi=300)
    plt.show()
'''
