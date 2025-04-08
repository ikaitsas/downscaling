# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:53:46 2025

@author: yiann

Current SRTM extent in my computer: 44N-17W-34S-32E
Regions outside this bracket will raise errors

By default, the extent values refer to the centers of the grid
cells, following the ERA5 and ERA5-Land datasets comventions.
Merging this way will cause a shift in values, equal to half the 
resolution in every direction, meanign the true extent that GDAL 
crops is actually bigger than specified by half the resolution in
each direction.

Extent convention in code follows GDAL commands order (W-N-E-S)
Because ERA5 convention is N-W-S-E, the naming of the produced files
follows this order...

This crops the 1-3arcsecond native resolution DEM derived morphography
products. Aggregation should be applied to the specified extent files,
as to match ERA5 grid cell centers and DEM cell centers, and generally
match the centers of the grid cells for any downscaling resolution...
"""

import os
import subprocess
#import rasterio
#from osgeo import gdal
from pathlib import Path


nc_file_resolution = 0.1  #era5-land resolution
input_dem = "output.tif"
extent = [19.6, 41.8, 28.3, 35.8]  #W-N-E-S
#set True if extent corresponds to center of grid cells
reference_center = True
#set True if you want to expand out to all directions
#else it just shifts the DEM north and west, matching ERA5
expand_to_all_directions = True


dem_file_name = input_dem[:-4]
cwd = Path.cwd()
morphography = ["dem", "slope", "aspect"]

srtm_dir = os.path.join(cwd, "srtm", "void-filled-3arcsec")
existing_files = set(os.listdir(srtm_dir))

print(f'Extracting subregion {extent} W-N-E-S...\n')

for morphi in morphography:
    if morphi == "dem":
        srtm_file = f"{dem_file_name}.tif"
    else:
        srtm_file = f"{dem_file_name}-{morphi}.tif"
    
    srtm_path = os.path.join(srtm_dir, srtm_file)
    
    if srtm_file not in existing_files:
        print(f"Skipping {morphi}, file not found: {srtm_file}")
        continue
    
    
    extent_file = (
        f"{dem_file_name}-{morphi}-extent-"
        f"{extent[1]}n-{extent[0]}w-{extent[3]}s-{extent[2]}e"
        f".tif"
        )
    extent_path = os.path.join(cwd, "outputs", "tif", extent_file)

    print(f'Extracting {morphi}...')
    
    if reference_center == True:
        # subwindow to extract - in projected coordinates
        if expand_to_all_directions == True:
            subwindow = [
                extent[0]-nc_file_resolution/2, 
                extent[1]+nc_file_resolution/2, 
                extent[2]+nc_file_resolution/2, 
                extent[3]-nc_file_resolution/2
                ]
        else:
            subwindow = [
                extent[0]-nc_file_resolution/2, 
                extent[1]+nc_file_resolution/2, 
                extent[2]-nc_file_resolution/2, 
                extent[3]+nc_file_resolution/2
                ]
    else:
        subwindow = extent
         

    cmd_crop = [
        "gdal_translate", 
        "-projwin",
        f"{subwindow[0]}", 
        f"{subwindow[1]}", 
        f"{subwindow[2]}", 
        f"{subwindow[3]}",
        "-projwin_srs", 
        "EPSG:4326",
        srtm_path,
        extent_path
        ]
    
    run_crop = subprocess.run(cmd_crop, 
                            capture_output=True, 
                            text=True)
    print("gdalbuildvrt Standard Output:", run_crop.stdout)
    print("gdalbuildvrt Standard Error:", run_crop.stderr)


#%% Python GDAL methodology - somehow crashes on the lapapeio machine...
'''
    try:
        gdal.UseExceptions()
        options = gdal.TranslateOptions(
            projWin = subwindow,
            projWinSRS = "EPSG:4326"
            )
        gdal.Translate(destName=extent_path, srcDS=srtm_path, 
                       options=options)
        
        print(f'Saving as:  {extent_file}')
        print('Following ERA5 extent convention...')
        print(f'Saving in:  {extent_path}\n')
    except Exception as e:
        print(f"Error:\n{e}")


ds = gdal.Open(extent_path)
gt = ds.GetGeoTransform()
rows, cols = ds.RasterYSize, ds.RasterXSize

ds = None
'''




