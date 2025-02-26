# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:43:53 2025

@author: yiann
"""

import os
from osgeo import gdal
from pathlib import Path


target_resolution = 0.1  # in degrees

input_dem = "output.tif"
extent = [21, 39, 24, 36]  #W-N-E-S
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
    
    ds = gdal.Open(tif_path)
    
    gt = ds.GetGeoTransform()

    x_res = int((gt[1] * ds.RasterXSize) / target_resolution)
    y_res = int((abs(gt[5]) * ds.RasterYSize) / target_resolution)
    
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
