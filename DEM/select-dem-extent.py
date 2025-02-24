# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:53:46 2025

@author: yiann

Current SRTM extent in my computer: 44N-17W-34S-32E
Regions outside this bracket will raise errors
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

cmd_crop = [
    "gdal_translate", "-projwin",
    f"{extent[0]-nc_file_resolution/2}", 
    f"{extent[1]+nc_file_resolution/2}", 
    f"{extent[2]+nc_file_resolution/2}", 
    f"{extent[3]-nc_file_resolution/2}",
    "-projwin_srs", "EPSG:4326",
    #"-projwin_srs EPSG:4326",  #produces error runnign from here
    #even though from the cmd this argument does not... whatever...
    srtm_path,
    extent_path
    ]

run_crop = subprocess.run(cmd_crop, 
                        capture_output=True, 
                        text=True)
print("gdalbuildvrt Standard Output:", run_crop.stdout)
print("gdalbuildvrt Standard Error:", run_crop.stderr)



