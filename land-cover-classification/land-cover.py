# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:15:10 2025

@author: yiann

CDR and ICDR Sentinel-3 Land Cover classifications on CDS

guides and specifications (v2.1 & v2.0 respectively) at:
https://dast.copernicus-climate.eu/documents/satellite-land-cover/WP2-FDDP-LC-2021-2022-SENTINEL3-300m-v2.1.1_PUGS_v1.1_final.pdf
https://dast.copernicus-climate.eu/documents/satellite-land-cover/D3.3.11-v1.0_PUGS_CDR_LC-CCI_v2.0.7cds_Products_v1.0.1_APPROVED_Ver1.pdf

the classes need a bit of merging and cleaning up...
added to the todo-list (in 2025-02-17)
extent is N-W-S-E
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


extent = [39, 24, 36, 21]
years = list(range(1992,2023))
single_year = 1992


#%% extracting file location - NEED TO MAKE THIS A FUNCTION
extent_string =  f'{extent[0]}.{extent[1]}.{extent[-2]}.{extent[-1]}'

subfolder = (
    f'area-subset.{extent_string}'
    )
folder = os.path.join("data", subfolder)
os.makedirs(folder, exist_ok=True)

year = single_year
if year > 2015:
    version = "v2.1.1"
    convention = "C3S"
else:
    version = "v2.0.7cds"
    convention = "ESACCI"

filename = (
    f'{convention}-LC-L4-LCCS-Map-300m-P1Y-{year}-{version}.area-subset.{extent_string}'
    f'.nc'
    )
file_path = os.path.join(folder, filename)


#%% xarray
ds = xr.open_dataset(file_path)

print(ds.variables)
print('\n')
print(ds.coords)
print('\n')
print('Variables: {}'.format(list(ds.keys())))
print('\n')
print('Coordinates: {}'.format(list(ds.coords)))

lc = ds.lccs_class.to_numpy()

#ds.close()



#%% visualization - make adjustments for choosing year
rgb_image  = np.zeros((lc.shape[1], lc.shape[2], 3), dtype=np.uint8)
class_color_mapping  = {
    # make colorings for 11,12,61,122,153,201
    
    0: (0, 0, 0),           #No Data 
    10: (255, 255, 100),    #Cropland, rainfed
    11: (255, 255, 100),    #Cropland, rainfed - 
    12: (255, 255, 100),    #Cropland, rainfed - 

    20: (170, 240, 240),    #Cropland, irrigated or post-flooding
    30: (220, 240, 100),    #Mosaic cropland (>50%) / natural vegetation (tree, shrub,herbaceous cover) (<50%)
    40: (200, 200, 100),    #Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) /cropland (<50%
    50: (0, 100, 0),        #Tree cover, broadleaved, evergreen, closed to open (>15%) 
    60: (0, 160, 0),        #Tree cover, broadleaved, deciduous, closed to open (>15%)
    61: (0, 160, 0),        #Tree cover, broadleaved, deciduous, closed to open (>15%)

    70: (0, 60, 0),         #Tree cover, needleleaved, evergreen, closed to open (>15%)
    80: (40, 80, 0),        #Tree cover, needleleaved, deciduous, closed to open (>15%)
    90: (120, 130, 0),      #Tree cover, mixed leaf type (broadleaved and needleleaved)
    100: (140, 160, 0),     #Mosaic tree and shrub (>50%) / herbaceous cover (<50%) 
    110: (190, 150, 0),     #Mosaic herbaceous cover (>50%) / tree and shrub (<50%) 
    120: (150, 100, 0),     #Shrubland
    122: (150, 100, 0),     #Shrubland - 

    130: (255, 180, 50),    #Grassland
    140: (255, 220, 210),   #Lichens and mosses
    150: (255, 235, 175),   #Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
    153: (255, 235, 175),   #Sparse vegetation (tree, shrub, herbaceous cover) (<15%)

    160: (0, 120, 90),      #Tree cover, flooded, fresh or brackish water 
    170: (0, 150, 120),     #Tree cover, flooded, saline water 
    180: (0, 220, 130),     #Shrub or herbaceous cover, flooded, fresh/saline/brackish water 
    190: (195, 20, 0),      #Urban areas
    200: (255, 245, 215),   #Bare areas 
    201: (255, 245, 215),   #Bare areas 

    210: (0, 70, 200),      #Water bodies 
    220: (255, 255, 255),   #Permanent snow and ice 
    
}

for class_code, color in class_color_mapping.items():
    mask = lc[0,:,:] == class_code
    for i in range(3):  # Assign RGB channels
        rgb_image[:, :, i][mask] = color[i]

plt.imshow(rgb_image, origin='upper', aspect='equal')
#plt.savefig(f'land-cover-{year}-area-subset.{extent_string}.png', dpi=2000)
plt.show()




