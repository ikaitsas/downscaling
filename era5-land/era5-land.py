# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:42:11 2025

@author: yiann

Tidying up ERA5-Land data and import DEM-derived morphography metadata 
in the respective coordinates

Will probably add land cover data in here too
And any other data needed, either as keys, coordinates, dimensions,
straight to the Datasets and DataArrays...
"""

print('Importing libraries and data...')
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


extent = [39, 21, 36, 24]
years = list(range(1992,2023))
timescale = 'monthly'
visualize = True
save_to_device = True


# morphography files
dem = xr.open_dataset("output-morphography-0.1deg.nc")
demHD = xr.open_dataset("output-morphography-0.01deg.nc")


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

print('Merging with the ERA5-Land resolution Morphography...')
ds.coords["dem"] = (["latitude", "longitude"], dem.dem.data)
ds.coords["dem"].attrs["units"] = "meters"
ds.coords["dem"].attrs["description"] = "Elevation at each lat-lon pair"

ds.coords["slope"] = (["latitude", "longitude"], dem.slope.data)
ds.coords["slope"].attrs["units"] = "degrees"
ds.coords["slope"].attrs["description"] = "Slope at each lat-lon pair"

ds.coords["aspect"] = (["latitude", "longitude"], dem.aspect.data)
ds.coords["aspect"].attrs["units"] = "degrees - offset from North"
ds.coords["aspect"].attrs["description"] = "Aspect at each lat-lon pair"

ds.coords["valid_year"] = (["valid_time"], ds.valid_time.dt.year.data)
ds.coords["valid_month"] = (["valid_time"], ds.valid_time.dt.month.data)

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

# explicitly order the columns, important for modelling!
t2mColumn = ds.t2m.name  
column_to_keep = [t2mColumn, 
                  ds.latitude.name, 
                  ds.longitude.name,
                  ds.dem.name, 
                  ds.slope.name,
                  ds.aspect.name,
                  ds.valid_year.name,
                  ds.valid_month.name
                  ]

if ("valid_time" in df.index.names) & ("valid_month" not in df.columns):
    print("f")
    df["valid_time"] = df.index.get_level_values("valid_time")
    df['valid_year'] = df.valid_time.dt.year
    df['valid_month'] = df.valid_time.dt.month

if "latitude" in df.index.names:
    df['latitude'] = df.index.get_level_values("latitude")      #lvl-1
if "longitude" in df.index.names:
    df['longitude'] = df.index.get_level_values("longitude")    #lvl-2

if "valid_time" in df.columns:
    df = df.drop(columns=['valid_time'])

df = df[column_to_keep]  #ensure t2m is always first column
#df = df.reset_index(drop=True)


#%% create downscalign HD array
scaling_factor = 10  #divide resolution by this number

print('Producing HD version...')
print(f'Dividing each grid cell {scaling_factor}x{scaling_factor} times...')
#axis=1 latitude, axis=2 longitude, change accordingly
#will probably add index extraction from Dataset dimensions
t2mHD_array = np.repeat(
    np.repeat(t2m_array, scaling_factor, axis=1), scaling_factor, axis=2)


scaled_coords = {
    #kane kalyterh diatypwsh sthn epilogh twn min/max sta lat/lon
    'valid_time': ds.t2m.valid_time.values,
    'latitude': demHD.latitude.data,
    'longitude': demHD.longitude.data,  
}

# better to do this a dataset, for consistency...
t2mHD = xr.DataArray(t2mHD_array,
                     coords=scaled_coords,
                     dims=["valid_time", "latitude", "longitude"]
                     )

t2mHD.coords["dem"] = (["latitude", "longitude"], demHD.dem.data)
t2mHD.coords["dem"].attrs["units"] = "meters"
t2mHD.coords["dem"].attrs["description"] = "Elevation at each lat-lon pair"

t2mHD.coords["slope"] = (["latitude", "longitude"], demHD.slope.data)
t2mHD.coords["slope"].attrs["units"] = "degrees"
t2mHD.coords["slope"].attrs["description"] = "Slope at each lat-lon pair"

t2mHD.coords["aspect"] = (["latitude", "longitude"], demHD.aspect.data)
t2mHD.coords["aspect"].attrs["units"] = "degrees - offset from North"
t2mHD.coords["aspect"].attrs["description"] = "Aspect at each lat-lon pair"

t2mHD.coords["valid_year"] = (["valid_time"], ds.valid_time.dt.year.data)
t2mHD.coords["valid_month"] = (["valid_time"], ds.valid_time.dt.month.data)


print(f'Extracting {era5Land_resolution/scaling_factor}deg HD dataframe...')
dfHD = t2mHD.to_dataframe(name='t2m')


if ("valid_time" in dfHD.index.names) & ("valid_month" not in dfHD.columns):
    print("f")
    dfHD['valid_time'] = dfHD.index.get_level_values("valid_time")  #lvl-0
    dfHD['valid_year'] = dfHD.valid_time.dt.year
    dfHD['valid_month'] = dfHD.valid_time.dt.month

if "latitude" in dfHD.index.names:
    dfHD['latitude'] = dfHD.index.get_level_values("latitude")      #lvl-1
if "longitude" in dfHD.index.names:
    dfHD['longitude'] = dfHD.index.get_level_values("longitude")    #lvl-2

if "valid_time" in dfHD.columns:
    dfHD = dfHD.drop(columns=['valid_time'])

dfHD = dfHD[column_to_keep]
#dfHD = dfHD.reset_index(drop=True)


#%% exportation to device
if save_to_device == True:
    # HD is for high resolution auxilliary variables
    # t2mHD is not "HD" by itself, just repeated
    
    df.to_parquet("df.parquet")
    ds.to_netcdf("t2m.nc")
    
    dfHD.to_parquet("dfHD.parquet")
    t2mHD.to_netcdf("t2mHD.nc")
    

# Bring data to a tensor format
# Create a CNN channel-based feature map
# ilithios chatgtp tropos - tha to kanw me diko mou
# alla se allh fash
t2mLD = np.expand_dims(ds.t2m.to_numpy(), axis=-1)

demHD_array = demHD.dem.to_numpy()
demHD_array = np.repeat( 
    np.expand_dims(demHD_array, axis=0), 
    ds.valid_time.shape[0], axis=0
    ) 
demHD_array = np.expand_dims(demHD_array, axis=-1)

slopeHD = np.repeat( 
    np.expand_dims(
        demHD.slope.values, 
        axis=0), 
    ds.valid_time.shape[0], axis=0
    )
slopeHD = np.expand_dims(slopeHD, axis=-1)

aspectHD = np.repeat( 
    np.expand_dims(
        demHD.aspect.to_numpy(), 
        axis=0), 
    ds.valid_time.shape[0], axis=0
    )
aspectHD = np.expand_dims(aspectHD, axis=-1)

lon, lat = np.meshgrid(demHD.longitude.values, demHD.latitude.values)

lon = np.repeat(
    np.expand_dims(lon, axis=0), 
    ds.valid_time.shape[0], axis=0
    )
lon = np.expand_dims(lon, axis=-1)

lat = np.repeat( 
    np.expand_dims(lat, axis=0), 
    ds.valid_time.shape[0], axis=0
    )
lat = np.expand_dims(lat, axis=-1)

months = np.expand_dims(ds.valid_month.to_numpy(), axis=-1)
years = np.expand_dims(ds.valid_year.to_numpy(), axis=-1)

hd_aux = np.concatenate(
    [lat, lon, demHD_array, slopeHD, aspectHD], 
    axis=-1
    )

time_aux = np.concatenate([years, months], axis=-1)

if save_to_device == True:
    np.save("t2mLD-input.npy", t2mLD)
    np.save("hd-auxilliary.npy", hd_aux)
    np.save("time-auxilliary.npy", time_aux)
# use t2mLD, hd_aux, time_aux for neural network building
# and training - the rest are in some chatgtp convo...
    

#%%
valid_time_index = 11
degree_spacing = 0.1


if visualize == True:
    t2m = ds.t2m - 273.15
    print('Mapping...')

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.LAKES.with_scale("50m"), 
                   edgecolor="black")
    
    #try the following with plt.pcolormesh() too
    #it just needs longitude, latitude, 2d-array
    #xr.plot assumes the center of the grid cell
    #plt.imshow assumes the top-left corner
    t2m.isel(valid_time=valid_time_index).plot(
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
    ax.set_title(f"Temperature {np.datetime_as_string(ds.valid_time.values[valid_time_index], unit='M')}")
    #ax.set_title('Tem')
    #plt.savefig('images-maps\\t2m-era5-land-cartopy.png', dpi=1000)
    plt.show()
    
    #t2m = None


print('Done')


