# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:53:05 2025

@author: yiann

NOTE: slope and aspect calculations require a metric projection!

if DEM spans multiple UTM zones, use EPSG:6933 (NSIDC EASE-Grid 2.0 Global)

for a more refined approach, apply a different UTM reprojection to every
DEM piece in a different UTM zone, e.g. EPSG:32634 (western Greece), and 
EPSG:32635 (eastern Greece), and then merge those different slope and 
aspect tiffs - generally 326XX for N, 327XX for S hemispheres
this clipping and merging is a quite tiresome prosedure though...

Step 1: Identify UTM Zones for the DEM
gdalinfo input_dem.tif

Step 2: Clip DEM into UTM Zone-Based Regions
gdal_translate -projwin xmin1 ymin1 xmax1 ymax1 input_dem.tif dem_utm_zone_1.tif
gdal_translate -projwin xmin2 ymin2 xmax2 ymax2 input_dem.tif dem_utm_zone_2.tif

Step 3: Reproject Each Tile to Its Correct UTM Zone
gdalwarp -t_srs EPSG:32632 -r bilinear dem_utm_zone_1.tif dem_utm_zone_1_utm.tif
gdalwarp -t_srs EPSG:32633 -r bilinear dem_utm_zone_2.tif dem_utm_zone_2_utm.tif

Step 4: Merge the Reprojected Tiles
gdal_merge.py -o dem_utm_merged.tif dem_utm_zone_1_utm.tif dem_utm_zone_2_utm.tif

other viable options for regions away from the poles are EPSG:3395 (Mercator) 
and EPSG:54009 (Mollweide Equal Area)

For continent-wide DEMs, consider using a projection designed for large regions:
Region	Projection	EPSG Code
Europe	Lambert Azimuthal Equal-Area	EPSG:3035
North America	Albers Equal-Area	EPSG:102003
South America	South America Albers	EPSG:102033
Africa	Africa Albers Equal-Area	EPSG:102022
Asia	Asia North Lambert Conformal Conic	EPSG:102025

reprohection to EPSG:4326 (Plate Caree) is applied at the end

coco.tif is 36-39N, 21-24E, so EPSG:32634 is used


----------------------------------WARNING--------------------------------------
the above distort the shae of the output tif files, the ydo not match the
original coco.tif shape. this will cause problems of compatibility for
the data blending required by the machine learning algorithm
this reprojecting back and forth is producing discrepancies, both in the
UTM reprojection, and the backwards LAT/LON projection, so this attempt is
by all means considered unsuccsessful

just go by the regular scaling approach for slope calculations
but do every DEM puzzle pieace one by one, since the scaling factor is
affected by latitude, and it has to be constant fo all of the DEM size
(the above is purely speculation - i hope there is a way to input a matrix
 as a scaling factor...), then merge the different slope pieaces together...
just a thought...
maybe even better split every 1x1deg piece to 0.2x1deg, at latitude

cos = np.cos(np.deg2rad(np.arange(39,35.9,-0.2)))
cosd = np.cos(np.deg2rad(np.arange(39,35.9,-0.2)))-np.cos(np.deg2rad(37.5))
cosdp = np.cos(np.deg2rad(np.arange(39,35.9,-0.2)))/np.cos(np.deg2rad(37.5))-1

im still not sure why aspect calculation does not need scaling in EPSG:4326...
will look more into it...
----------------------------------WARNING--------------------------------------

"""

import os
import numpy as np
import subprocess
import logging
from multiprocessing import Pool
from osgeo import gdal

# Input DEM
input_dem = "coco.tif"


reproject = False  #keep it that way, else it changes raster shapes
plotSlopeAspect = False
# Set up logging
logging.basicConfig(filename="process.log", 
                    level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")


# Determine UTM Zone (Assume Central Longitude for UTM Projection)
def get_utm_epsg(longitude):
    zone = int((longitude + 180) / 6) + 1
    epsg_code = 32600 + zone if longitude >= 0 else 32700 + zone  
    # 326XX for North, 327XX for South
    return epsg_code


gdal.UseExceptions()

print('Importing DEM FIle...')
ds = gdal.Open(input_dem)
if ds is None:
    raise ValueError("Something went wrong!")
'''
if ds is None:
    logging.error("Error opening DEM file!")
    exit(1)
'''   
gt = ds.GetGeoTransform()
res_x, res_y = gt[1], -gt[5]  # Pixel resolution (negative for y)
xmin, ymax = gt[0], gt[3]  # Top-left coordinates
xmax = xmin + res_x * ds.RasterXSize  # Bottom-right longitude
ymin = ymax - res_y * ds.RasterYSize  # Bottom-right latitude

center_lon = (xmin + xmax) / 2
center_lat = (ymin + ymax) / 2

scale_factor = 111320 / np.cos(np.radians(center_lat))



slope_ll = "cocoSlope.tif"
print('Computing Slope...')
subprocess.run([
    "gdaldem", "slope", input_dem, slope_ll, "-s", f"{scale_factor}"
], check=True)

aspect_ll = "cocoAspect.tif"  #name despite no scaling...
print('Computing Aspect...')
subprocess.run([
    "gdaldem", "aspect", input_dem, aspect_ll
], check=True)

# Print output files   
print("\nProcessing Complete!")
print('Final (EPSG:4326) Files:')
print(f"- Slope (EPSG:4326): {slope_ll}")
print(f"- Aspect (EPSG:4326): {aspect_ll}")

    
#%% reproject the DEM to a metric UTM space - NOT RECOMMENDED
if reproject == True:
    utm_epsg = get_utm_epsg(center_lon)
    #logging.info(f"Detected UTM Zone: EPSG:{utm_epsg}")
    print(f'Detected UTM Zone: EPSG:{utm_epsg}')
    
    # Step 1: Reproject DEM to UTM
    utm_dem = "coco_utm.tif"
    #logging.info("Reprojecting DEM to UTM...")
    print("Reprojecting DEM to UTM...")
    subprocess.run([
        "gdalwarp", "-t_srs", f"EPSG:{utm_epsg}", "-r", "near",
        #"-tr", str(res_x), str(res_y),  # Preserve exact resolution
        #"-te", str(xmin), str(ymin), str(xmax), str(ymax),  # Preserve exact extent
        #"-tap",  # Align pixels to the target grid
        #"ts", str(ds.RasterXSize), str(ds.RasterYSize),
        input_dem, utm_dem
    ], check=True)
    
    # Step 2: Compute Slope in UTM
    slope_utm = "cocoSlope_utm.tif"
    print('Computing Slope...')
    subprocess.run([
        "gdaldem", "slope", utm_dem, slope_utm, "-s", "1.0"
    ], check=True)
    
    # Step 3: Compute Aspect in UTM
    aspect_utm = "cocoAspect_utm.tif"
    print('Computing Aspect...')
    subprocess.run([
        "gdaldem", "aspect", utm_dem, aspect_utm
    ], check=True)
    
    # Step 4: Reproject Slope Back to EPSG:4326
    slope_ll = "cocoSlope_reproj.tif"
    print('Reprojecting Slope to EPSG:4326...')
    subprocess.run([
        "gdalwarp", "-t_srs", "EPSG:4326", "-r", "near",
         #"-tr", str(res_x), str(res_y),
         #"-te", str(xmin), str(ymin), str(xmax), str(ymax),
         #"-tap",  # Ensure pixel alignment
        slope_utm, slope_ll
    ], check=True)
    
    # Step 5: Reproject Aspect Back to EPSG:4326
    aspect_ll = "cocoAspect_reproj.tif"
    print('Reprojecting Aspect to EPSG:4326...')
    subprocess.run([
        "gdalwarp", "-t_srs", "EPSG:4326", "-r", "near",
         #"-tr", str(res_x), str(res_y),
         #"-te", str(xmin), str(ymin), str(xmax), str(ymax),
         #"-tap",  # Ensure pixel alignment
        aspect_utm, aspect_ll
    ], check=True)
    
    # Print output files
    print("\nProcessing Complete!")
    print('\nUTM Projected Files:')
    print(f"- Projected DEM (UTM): {utm_dem}")
    print(f'- Slope (EPSG:{utm_epsg}): {slope_utm})')
    print(f'- Aspect (EPSG:{utm_epsg}): {aspect_utm})')
    
    print('\nFinal (EPSG:4326) Reprojected Files:')
    print(f"- Slope (EPSG:4326): {slope_ll}")
    print(f"- Aspect (EPSG:4326): {aspect_ll}")

 





