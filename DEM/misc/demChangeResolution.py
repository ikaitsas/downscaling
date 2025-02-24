# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:41:27 2025

@author: yiann

gdal commands, such as "gdalwarp", "gdaldem" are used via OSGEO4W
install that first, and make nadditions to PATH if necassary
"""
# make this a function!

import subprocess
from osgeo import gdal
from pathlib import Path

input_dem = 'coco.tif'
# Output resolution must be in degrees
resolution = 0.01


output_dem = f"{input_dem[:-4]}{resolution}deg.tif"
# Delete the output file if it already exists
if Path(output_dem).exists():
    print(f'Deleted already existig file named {output_dem}...\n')
    Path(output_dem).unlink()


print(f'Performing Aggregation to {resolution} degrees...\n')
aggregation_command = [
    "gdalwarp", "-tr", str(resolution), str(resolution), "-r", "average",
    input_dem, output_dem
    ]

# or simply: subprocess.run( aggregation_command, check=True)
output_demAggregation = subprocess.Popen(
    aggregation_command, 
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )
for line in output_demAggregation.stdout:
    # Print output in real time
    print(line, end="")

print('Aggregation Complete.')
print(f'File produced: {output_dem}\n')


# Print output .tif file metadata
output_demINFO = subprocess.Popen(
    ["gdalinfo", output_dem],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )
for line in output_demINFO.stdout:
    # Print output in real time
    print(line, end="")  


'''
gdal.UseExceptions()
dataset = gdal.Open(output_dem, gdal.GA_ReadOnly)
geotransform = dataset.GetGeoTransform()
array = dataset.GetRasterBand(1).ReadAsArray()
'''

