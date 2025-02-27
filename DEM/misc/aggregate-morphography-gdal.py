# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:43:53 2025

@author: yiann

This can also be done by exporting each .tif to an array, and
performing aggregation there. Either way, the sxis of the latitude
and longitude values must be specified for .nc file exportation

There is a a=slight mismatch, probably a byproduct of the offset
that ERA5 data have and SRTM data need to match that.

I'll probably do a first aggregation at taret_resolution/4, and
then smooth things out, i'll see if that works in the future...

"""

import os
import numpy as np
from osgeo import gdal
from pathlib import Path


target_resolution = 0.1  # in degrees
input_dem = "output.tif"
extent = [21, 39, 24, 36]  #W-N-E-S
export_nc_to_device = True
input_dem_native_resolution = 0.00083333333  # 3arcsec

dem_file_name = input_dem[:-4]
cwd = Path.cwd()
morphography = ["dem", "slope", "aspect"]
outputs_dir = os.path.join(cwd, "outputs", "tif")
existing_files = set(os.listdir(outputs_dir))


for morphi in morphography:
    tif_file = (
        f"{dem_file_name}-{morphi}-extent-"
        f"{extent[1]}n-{extent[0]}w-{extent[3]}s-{extent[2]}e"
        f".tif"
        )
    tif_path = os.path.join(cwd, "outputs", "tif", tif_file)
    
    aggregated_file = (
        f"{dem_file_name}-{morphi}-extent-"
        f"{extent[1]}n-{extent[0]}w-{extent[3]}s-{extent[2]}e"
        f"-{target_resolution}deg.tif"
        )
    aggregated_path = os.path.join(cwd, "outputs", "tif", aggregated_file)
    
    gdal.UseExceptions()
    
    
    # can use the _res variables, for explicit pixel fiting
    # in the new aggregated grid
    options = gdal.WarpOptions(
        xRes=target_resolution,
        yRes=target_resolution,  
        resampleAlg="average",
        format="GTiff"
        )
    try:
        gdal.Warp(
            aggregated_path,
            tif_path,
            options=options
            )
    except Exception as e:
        print('Error:\n{e}')


    if export_nc_to_device == True:
        ds = gdal.Open(aggregated_path)
        
        gt = ds.GetGeoTransform()
        rows, cols = ds.RasterYSize, ds.RasterXSize
        x_min = gt[0] #+ input_dem_native_resolution/2
        y_max = gt[3] #- input_dem_native_resolution/2
        x_res = gt[1]
        y_res = gt[5]
        
        latitudes = np.linspace(y_max, y_max + y_res * rows, rows)
        longitudes = np.linspace(x_min, x_min + x_res * cols, cols)
        
        latitudes_centered = latitudes + y_res/2
        latitudes_centered = np.round(latitudes_centered/target_resolution)*target_resolution
        longitudes_centered = longitudes + x_res/2
        longitudes_centered = np.round(longitudes_centered/target_resolution)*target_resolution

        
        #array = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        
        
        
        
        