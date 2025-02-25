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
"""

import os
import subprocess
from pathlib import Path


nc_file_resolution = 0.1  #era5-land resolution
extent = [21, 39, 24, 36]  #W-N-E-S
#set True if extent corresponds to center of grid cells
reference_center = True


cwd = Path.cwd()
srtm_path = os.path.join(cwd, "srtm", "void-filled-3arcsec", "output.tif")

extent_file = (
    f"dem-extent-"
    f"{extent[1]}N-{extent[0]}W-{extent[3]}S-{extent[2]}E"
    f".tif"
    )
extent_path = os.path.join(cwd, "outputs", "tif", extent_file)

print(f'Extracting subregion {extent} W-N-E-S...\n')
print(f'Saving as:  {extent_file}\nFollowing ERA5 extent convention...')
print(f'Saving in:  {extent_path}')


#better do it with gdal.Translate() & gdal.TranslateOptions()
#it will be mush clearer this way
if reference_center == True:
    cmd_crop = [
        "gdal_translate", 
        "-projwin",
        f"{extent[0]-nc_file_resolution/2}", 
        f"{extent[1]+nc_file_resolution/2}", 
        f"{extent[2]+nc_file_resolution/2}", 
        f"{extent[3]-nc_file_resolution/2}",
        "-projwin_srs", 
        "EPSG:4326",
        srtm_path,
        extent_path
        ]
else:  #for regular GDAL cropping
    cmd_crop = [
        "gdal_translate", 
        "-projwin",
        f"{extent[0]}", 
        f"{extent[1]}", 
        f"{extent[2]}", 
        f"{extent[3]}",
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



