# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:45:08 2025

@author: yiann
"""

import os
import glob
import subprocess
from pathlib import Path


cwd = Path.cwd()
tif_files = glob.glob(os.path.join(cwd, "*.tif"))
# mediorce try to only track SRTM tiles, not outputs
tif_files = [
    file for file in tif_files 
    if os.path.basename(file).startswith('n')
    ]

if not tif_files:
    print("No .tif files found in the directory.")
    
else:
    # Output VRT file
    print('Merging tifs...')
    print('Building Virtual Dataset...')
    output_vrt = os.path.join(cwd, "output.vrt")

    cmd_virtual = ["gdalbuildvrt", output_vrt] + tif_files
    
    result_virtual = subprocess.run(cmd_virtual, 
                            capture_output=True, 
                            text=True)
    print("gdalbuildvrt Standard Output:", result_virtual.stdout)
    print("gdalbuildvrt Standard Error:", result_virtual.stderr)
    
    print('\nTranslating to a .tif file...')
    output_tif = os.path.join(cwd, "output.tif")
    
    cmd_translate = ["gdal_translate", output_vrt, output_tif]
    
    result_translate = subprocess.run(cmd_translate, 
                                      capture_output=True, 
                                      text=True)
    print("gdal_translate Output:", result_translate.stdout)
    print("gdal_translate Error:", result_translate.stderr)








