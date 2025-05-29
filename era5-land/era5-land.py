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

from dask.dataframe import from_pandas

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator


extent = [41.8, 19.6, 35.8, 28.3] #N-W-S-E
years = list(range(1992,2023))
timescale = 'monthly'
visualize = True
save_to_device = True
extract_nn_training_data = False  #keep false!

downscaling_year_start = 2017
target_resolution = 0.01
coarse_resolution = 0.1


# morphography files
dem = xr.open_dataset(f"output-morphography-{coarse_resolution}deg.nc")
demHD = xr.open_dataset(f"output-morphography-{target_resolution}deg.nc")

lc = xr.open_dataarray(f"land-cover-{coarse_resolution}deg.nc")
lcHD = xr.open_dataarray(f"land-cover-{target_resolution}deg.nc")


scaling_factor = coarse_resolution/target_resolution


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

#ds = ds.drop_isel(latitude=-1, longitude=-1)

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

lc = lc.astype("uint8")
lc = lc.rename(year='valid_year').sel(valid_year=ds['valid_year'])
lc = lc.assign_coords(
    latitude=ds.latitude,
    longitude=ds.longitude
) # eliminate various float point mismatches
ds.coords["land_cover"] = lc

lc = None
    

'''
# care, doing:
ds.coords["land_cover"] = (("valid_year", "latitude", "longitude"), lc.data)
# will leave the land_cover coordinate unaligned with the keys of ds, meaning
# that it wont be able to be directly mapped in case of selections, because
# valid_time != valid_year, alignment must be done explicitly, using .sel or
# .interp
'''

if "number" in ds.coords:
    ds = ds.reset_coords(["number"], drop=True)
if "expver" in ds.coords:
    ds = ds.reset_coords(["expver"], drop=True)


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
                  ds.land_cover.name,
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

if save_to_device == True:
    df.to_parquet(f"df-{coarse_resolution}deg.parquet")
    ds.t2m.to_netcdf(f"t2m-{coarse_resolution}deg.nc")

#df = df.reset_index(drop=True)
#ddf = from_pandas(df, int(ds.valid_month.values.shape[0]/2))
df = None


#%% create downscalign HD array
print('Producing HD version...')
print(f'Dividing each grid cell {scaling_factor}x{scaling_factor} times...')
#axis=1 latitude, axis=2 longitude, change accordingly
#will probably add index extraction from Dataset dimensions
t2m_array = ds.t2m.to_numpy()
hd_t2m_array = np.repeat(
    np.repeat(t2m_array, scaling_factor, axis=1), scaling_factor, axis=2)


scaled_coords = {
    #kane kalyterh diatypwsh sthn epilogh twn min/max sta lat/lon
    'valid_time': ds.t2m.valid_time.values,
    'latitude': demHD.latitude.data,
    'longitude': demHD.longitude.data,  
}

# better to do this a dataset, for consistency...
t2mHD = xr.DataArray(hd_t2m_array,
                     coords=scaled_coords,
                     dims=["valid_time", "latitude", "longitude"]
                     )
hd_t2m_array = None
t2m_array = None

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

lcHD = lcHD.astype("uint8")
lcHD = lcHD.rename(year='valid_year').sel(valid_year=t2mHD['valid_year'])
lcHD = lcHD.assign_coords(
    latitude=t2mHD.latitude,
    longitude=t2mHD.longitude
) # eliminate various float point mismatches
t2mHD.coords["land_cover"] = lcHD

lcHD = None    

'''
# Keep only part for downscaling - dont use this in training
t2mHD = t2mHD.sel(
    valid_time=t2mHD.valid_time[t2mHD.valid_year >= downscaling_year_start]
    )
'''



#%% extract as arrays- might be applicable for large datasets
# MAKE SURE THESE ARE PERFECTLY ALIGNED, LETS SAY C-STYLE (DEFAULT)
'''
time_levels = t2mHD.valid_time.shape[0]

X_flat = t2mHD.values.reshape(time_levels,-1)

hd_months = t2mHD.valid_month.values.astype(np.int8)
hd_years = t2mHD.valid_year.values.astype(np.int16)

hd_latitude = t2mHD.latitude.values.astype(np.float32)
hd_longitude = t2mHD.longitude.values.astype(np.float32)
hd_dem = np.tile(t2mHD.dem.values.flatten(), hd_months.shape).astype(np.int16)
hd_slope = np.tile(t2mHD.slope.values.flatten(), hd_months.shape).astype(np.float32)
hd_aspect = np.tile(t2mHD.aspect.values.flatten(), hd_months.shape).astype(np.int16)
'''


#%% convert to dataframe - not applicable for very large datasets
print(f'Extracting {coarse_resolution/scaling_factor}deg HD dataframe...')
# since, i think, the fitting extra trees algorithm does not take into account
# the year, the HD version of the geographic area can be taken for 12 months,
# as only the month is the temporal covariate needed.
# the HD dataframe can contain only a year's wort hof downscaled data.
# residual downscaling can be done on a different dataset/dataarray.
'''
dfHD = t2mHD.stack(
    points=("latitude", "longitude")
    ).to_series().reset_index(name="t2m")  # needs a lot of memeory...
'''
#dfHD = t2mHD.to_dataframe(name='t2m') # need a lot of memory...

# since extracting the whole thing at once is memory intensive, i will 
# extract a few years at a time as a dataframe, and use them later...
valid_time = t2mHD.valid_time.to_index()

years_per_file = 5
start_years = range(valid_time[0].year, valid_time[-1].year + 1, years_per_file)

for start_year in start_years:
    end_year = start_year + years_per_file - 1
    if end_year > valid_time[-1].year:
        end_year = valid_time[-1].year
    print(f"Extracting Period {start_year}-{end_year} as Dataframe...")
    # Select time slice
    mask = (valid_time.year >= start_year) & (valid_time.year <= end_year)
    t2mHD_subset = t2mHD.sel(valid_time=valid_time[mask])

    # Skip if no data in this range
    if t2mHD_subset.valid_time.size == 0:
        continue

    # can optimize further with:
    # .stack(points=("latitude", "longitude")).to_series()) ??
    dfHD = t2mHD_subset.to_dataframe(name="t2m")#.reset_index()
    
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

    #dfHD = dfHD.loc[:, column_to_keep]

    if save_to_device == True:
        filename = f"df-{start_year}-{end_year}-{target_resolution}deg.parquet"
        dfHD.loc[:, column_to_keep].to_parquet(filename)
        print(f"Saved: {filename}")


if save_to_device == True:
    # HD is for high resolution auxilliary variables
    # t2mHD is not "HD" by itself, just repeated
    t2mHD.to_netcdf(f"t2m-{target_resolution}deg.nc")


#%%
valid_time_index = 341
degree_spacing = 0.5


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
        cbar_kwargs={"label": "Temperature [°C]"},
        vmin=0, 
        vmax=26
    )
    
    ax.coastlines(resolution="10m", linewidth=0.35)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.25)
    ax.add_feature(cfeature.LAND)
    
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.25, linestyle="--", color="gray"
        )
    gl.xlocator = MultipleLocator(degree_spacing)
    gl.ylocator = MultipleLocator(degree_spacing)
    gl.top_labels = False
    gl.right_labels = False
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
    timestamp_string = f"{np.datetime_as_string(ds.valid_time.values[valid_time_index], unit='M')}"
    ax.set_title(f"ERA5-Land 2m-Air Temperature - {timestamp_string}")
    #ax.set_title('Tem')
    #plt.savefig('images-maps\\t2m-era5-land-{timestamp_string}.png', dpi=500, bbox_inches="tight")
    plt.show()
    
    #t2m = None


print('Done')


#%% multiplot
valid_time_index = 341
degree_spacing = 0.5

if visualize == True:
    t2m = ds.t2m - 273.15
    print('Mapping2...')
    
    # order: t2mLD, t2mLD_pred, t2m_pred, t2m_res
    # dont change the order of the above
    # or change it everywhere the same below!
    timestamp_string1 = f"{np.datetime_as_string(ds.valid_time.values[valid_time_index], unit='M')}"
    timestamp_string2 = f"{np.datetime_as_string(ds.valid_time.values[valid_time_index+6], unit='M')}"
    titles = [
        f"ERA5-Land 2m-Air Temperature - {timestamp_string1}",
        f"ERA5-Land 2m-Air Temperature - {timestamp_string2}"
        ]
    
    mins = [
        np.nanmin(t2m.values[valid_time_index,:,:]),
        np.nanmin(t2m.values[valid_time_index+6,:,:]),
        ]
    maxes = [
        np.nanmax(t2m.values[valid_time_index,:,:]),
        np.nanmax(t2m.values[valid_time_index+6,:,:]),
        ]
    
    fig, axes = plt.subplots(
        2, 1,  #rows, columns
        figsize=(7, 10), 
        subplot_kw={'projection': ccrs.PlateCarree()}
        )
    
    vmin = np.floor( np.min(mins) )
    vmax = np.ceil( np.max(maxes) )
    
    for ax, da, title in zip(
            axes.flat, 
            [t2m[valid_time_index,:,:],t2m[valid_time_index+6,:,:]], 
            titles
            ):
        
        img = da.plot.pcolormesh(
            ax=ax, cmap='inferno', transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, add_colorbar=False
            )

        ax.coastlines(resolution="10m", linewidth=0.25)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.25)
        ax.add_feature(cfeature.LAND)
        ax.set_title(title)
        
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.25, linestyle="--", color="gray",
            alpha=0.5
            )
        gl.xlocator = MultipleLocator(degree_spacing)
        gl.ylocator = MultipleLocator(degree_spacing)
        gl.top_labels = False
        gl.right_labels = False
        
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.04, 0.78])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("Temperature [°C]")
        
    #plt.tight_layout() 
    #plt.savefig(f"images-maps\\t2m-era5-land-multiplot{valid_time_index}.png", dpi=1500, bbox_inches="tight")
    plt.show() 


#%% correlation between parameters
# nvm do it on the nan-cleaned df, not here


#%% exportation to device for NN training
'''
if extract_nn_training_data == True:
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
'''
