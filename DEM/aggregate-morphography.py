# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:43:53 2025

@author: yiann

kane auto to script synarthsh, jesus

might be best to aggregate first to 0.01 degrees and
then aggregate again to 0.1 degrees, especially for aspect??

and make the importation of files a bit more clear
ase ta perierga me to psaxnw se directories klp
oute esy den tha ta thimasai se ligo
"""

import os
import rasterio
import numpy as np
import xarray as xr
#from osgeo import gdal
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from sklearn.impute import KNNImputer
from skimage.measure import block_reduce


target_res = 0.1  # in degrees
input_dem = "output.tif"
extent = [19.6, 41.8, 28.3, 35.8]  #W-N-E-S
export_nc_to_device = True


dem_file_name = input_dem[:-4]
cwd = Path.cwd()
morphography = ["dem", "slope", "aspect"]
outputs_dir = os.path.join(cwd, "outputs", "tif")
existing_files = set(os.listdir(outputs_dir))


for i,morphi in enumerate(morphography):
    tif_file = (
        f"{dem_file_name}-{morphi}-extent-"
        f"{extent[1]}n-{extent[0]}w-{extent[3]}s-{extent[2]}e"
        f".tif"
        )
    tif_path = os.path.join(cwd, "outputs", "tif", tif_file)
    
    
    #gdal.UseExceptions()
    
    #ds = gdal.Open(tif_path)
    with rasterio.open(tif_path) as ds:
        array = ds.read(1)
    
    array = array.astype("float32")
    
    #array = ds.read(1, dtype="float32")
    
    #introduce coarse corrections for slope and aspect too
    #like the 
    if morphi == "dem":
        array[array<-100] = 0
    if morphi == "slope":
        array[array<0] = 0
        
    #------------------------------------------------------------
    #apo edw kai katw allakse ta gdal calls me rasterio calls
    gt = ds.transform
    rows, cols = ds.height, ds.width
    x_min = gt[2] #+ input_dem_native_resolution/2
    y_max = gt[5] #- input_dem_native_resolution/2
    x_res = gt[0]
    y_res = gt[4]
    
    latitudes = np.arange(y_max, y_max+rows*y_res, y_res)
    longitudes = np.arange(x_min, x_min+cols*x_res, x_res)
    #the above are for the native resolution
   
    
    print(f'Native resolution: {x_res*3600:.1f} arcseconds.')
    print(f'Target resolution: {target_res} degrees.')
    #add error raising if target_res%x_res!=0
    block_size = int(target_res/x_res)
    new_rows, new_cols = int(rows//block_size), int(cols//block_size)
    
    reshaped = array.reshape(new_rows, block_size, new_cols, block_size)
    
    array_agg = np.nanmean(reshaped, axis=(1, 3)).round()
    
    if morphi == "aspect":
        #aspect filtering done later, to pass the flat surfaces
        #info (-9999 values) better to the aggregation?
        #idk if this is better, ill look into it
        array_agg = np.nanmedian(reshaped, axis=(1, 3)).round()
        array[array<0] = -1
        array_agg[array_agg<0] = -1
    
    y_res_agg = -target_res
    x_res_agg = target_res
    
    # remove native resolution offset, # and center the grid cell value
    # and ensure no extra latitudes/longitudes were computed
    latitudes_agg = np.arange(
        y_max, y_max+new_rows*y_res_agg, y_res_agg
        ) + y_res/2 + y_res_agg/2
    latitudes_agg = latitudes_agg[:array_agg.shape[0]]
    longitudes_agg = np.arange(
        x_min, x_min+new_cols*x_res_agg, x_res_agg
        ) + x_res/2 + x_res_agg/2
    longitudes_agg = longitudes_agg[:array_agg.shape[1]]
    
    if export_nc_to_device == True:
        # initialize Dataset
        if  i == 0:
            dims = [
                "latitude",
                "longitude"
                ]
            
            ds_nc = xr.Dataset(
                coords = {
                    "latitude" : latitudes_agg,
                    "longitude" : longitudes_agg
                    },
            )
        
        ds_nc[morphi] = (dims, array_agg)
        
        
        

netcdf_name = f"{dem_file_name}-morphography-{target_res}deg.nc"
netcdf_path = os.path.join(cwd, "outputs", "python", netcdf_name)
ds_nc.to_netcdf(netcdf_path)


#%% optikopoihsh
for morphi in morphography:
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.LAKES.with_scale("50m"), 
                   edgecolor="black")
    
    if morphi == "dem":
        cmap=plt.cm.inferno_r
        label = "Elevation [m]"
        title = f'{morphi.upper()} - {target_res}°'
        vmin, vmax = 0, np.nanmax(ds_nc[morphi].values)
        
    if morphi == "slope":
        cmap=plt.cm.magma_r
        label = "Slope [°]"
        title = f'{morphi.capitalize()} - {target_res}°'
        vmin, vmax = 0, np.nanmax(ds_nc[morphi].values)
        
    if morphi == "aspect":
        cmap=plt.cm.twilight
        label = "Aspect [°]"
        title = f'{morphi.capitalize()} - {target_res}°'
        vmin, vmax = -1, np.nanmax(ds_nc[morphi].values)
    
    #try the following with plt.pcolormesh() too
    #it just needs longitude, latitude, 2d-array
    #xr.plot assumes the center of the grid cell
    #plt.imshow assumes the top-left corner
    ds_nc[morphi].plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cmap=cmap, 
        cbar_kwargs={"label": label},
        vmin=vmin, vmax=vmax
    )
    
    ax.coastlines(resolution="10m", linewidth=0.25)
    ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.25)
    ax.add_feature(cfeature.LAND)
    
    gl = ax.gridlines(draw_labels=True, linestyle="--", 
                      linewidth=0.25, color="gray",
                      xlocs=np.arange(
                          extent[0], 
                          extent[2], 
                          0.3
                          ),  #or: mticker.FixedLocator
                      ylocs=np.arange(
                          extent[1], 
                          extent[3], 
                          -0.3
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
    ax.set_title(title)
    plt.savefig(f'outputs\\images\\{morphi}-{target_res}deg.png', 
                dpi=1000, bbox_inches="tight")
    plt.show()
    

#%% koments
'''
#very good aggregating method too, very readable
#bu it can thandle NaN values
array_agg1 = block_reduce(array, 
                          (block_size, block_size), 
                          np.mean).round()
'''
'''
    if export_nc_to_device == True:
        latitudes = np.linspace(y_max, y_max + y_res * rows, rows)
        longitudes = np.linspace(x_min, x_min + x_res * cols, cols)
        #produce another array for the aggregates, using different x_res,
        #y_res. the mismatches were most likely because of the linspace
        #numpy method, the correct one in this case is arange....
        #efaga wres gia authn thn paparia.....
        #np.arange also ensures correct spacing...
        
        latitudes_centered = latitudes + y_res/2
        latitudes_centered = np.round(
            latitudes_centered/target_resolution
            )*target_resolution
        longitudes_centered = longitudes + x_res/2
        longitudes_centered = np.round(
            longitudes_centered/target_resolution
            )*target_resolution

        
        #array = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        
'''     
'''
def aggregate_array_ignore_nan(array, block_size):
    """
    Aggregates a 2D numpy array by averaging non-overlapping blocks while 
    ignoring NaNs.
    
    Parameters:
    - arr: Input 2D numpy array (original high-resolution data)
    - block_size: Size of each aggregation block (e.g., 12 for 12x12 blocks)

    Returns:
    - Aggregated 2D numpy array (lower resolution)
    """
    rows, cols = array.shape
    new_rows, new_cols = rows // block_size, cols // block_size

    # Reshape into blocks
    reshaped = array[:new_rows * block_size, :new_cols * block_size].\
    reshape(new_rows, block_size, new_cols, block_size)

    # Compute nanmean instead of mean
    return np.nanmean(reshaped, axis=(1, 3))

def nanmean_block(block):
    return np.nanmean(block) if np.any(~np.isnan(block)) else np.nan
'''    
        
        