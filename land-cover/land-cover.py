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
extent is N-E-S-W

resolution is 10 arcseconds (0.0027778 degrees) - 36 grid cells for 0.1 degrees
of the era5-land grid cell

THE SCRIPT NEEDS MORE WORK
FIRST OF ALL, A FOR LOOP IS NEEDED, TO EXTRACT ALL THE YEARS. PREFERABLY INTO
A SINGLE .nc FILE
RIGHT NOW I DID THE EXTRACTION YEAR BY YEAR
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt


extent = [44, 17, 34, 32]
years = list(range(1992,2023))
single_year = 1992

target_resolution = 0.1
target_extent = [41.8, 19.6, 35.8, 28.3] #N-W-S-E

era5_resolution = 0.1  # resolution for era5-land
reference_center = True
expand_to_all_directions = True


#%% setting up the aggregated, aligned, cropped land cover
if reference_center == True:
    # subwindow to extract - in projected coordinates
    if expand_to_all_directions == True:
        subwindow = [
            target_extent[0]+era5_resolution/2, 
            target_extent[1]-era5_resolution/2, 
            target_extent[2]-era5_resolution/2, 
            target_extent[3]+era5_resolution/2
            ]
    else:
        # dont extend the last lat/lon element 
        subwindow = [
            target_extent[0]+era5_resolution/2, 
            target_extent[1]-era5_resolution/2, 
            target_extent[2]+era5_resolution/2, 
            target_extent[3]-era5_resolution/2
            ]
else:
    subwindow = target_extent


latitudes_subset = np.arange(
    subwindow[0],
    subwindow[2],
    -target_resolution
    ) -target_resolution/2

longitudes_subset = np.arange(
    subwindow[1],
    subwindow[3],
    target_resolution
    ) +target_resolution/2


#%% extracting file location - NEED TO MAKE THIS A FUNCTION
# epishmo
extent_string =  f'{extent[0]}.{extent[3]}.{extent[2]}.{extent[1]}'
# apo request
#extent_string =  f'N{extent[0]}-W{extent[1]}-S{extent[-2]}-E{extent[-1]}'


'''
# epishma
subfolder = (
    f'area-subset.{extent_string}'
    )
'''
# apo requests
subfolder = f'extent__{extent_string}'
folder = os.path.join("outputs", subfolder)
os.makedirs(folder, exist_ok=True)

year = single_year

#for year in years:
if year > 2015:
    version = "v2.1.1"
    convention = "C3S"
else:
    version = "v2.0.7cds"
    convention = "ESACCI"


# epishmh onomasia
filename = (
    f'{convention}-LC-L4-LCCS-Map-300m-P1Y-{year}-{version}.area-subset.{extent_string}'
    f'.nc'
    )

# onomasia apo requests
#filename = (f'land-cover__{extent_string}__Period{year}.nc')
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
    11: (255, 220, 90),    #Cropland, rainfed - 
    12: (255, 200, 100),    #Cropland, rainfed - trees?

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

# Print Class names
for class_name, class_value, class_color in zip(
        ds.lccs_class.attrs["flag_meanings"].split(),
        ds.lccs_class.attrs["flag_values"],
        ds.lccs_class.attrs["flag_colors"].split()
        ):
    print(f"{class_name} - {class_value}   ({class_color})")


# visualize the original land cover
# need to make the following code snippet a function, that takes
# as inputs the class_color_mapping, and the land cover array
for class_code, color in class_color_mapping.items():
    mask = lc[0,:,:] == class_code
    for i in range(3):  # Assign RGB channels
        rgb_image[:, :, i][mask] = color[i]

plt.imshow(rgb_image, origin='upper', aspect='equal')
plt.xticks([])
plt.yticks([])
#plt.savefig(f'land-cover-{year}-area-subset.{extent_string}.png', dpi=2000, bbox_inches="tight")
plt.show()


#%% simplification of land cover classes
lc_simpler = lc.copy()

# new codes for agriculture - 1
lc_simpler[lc_simpler==10] = 1
lc_simpler[lc_simpler==11] = 1  #herbaceous cropland
#lc_simpler[lc_simpler==12] = 10  #tree crops might need different treatment
lc_simpler[lc_simpler==20] = 1
lc_simpler[lc_simpler==30] = 1

# new code for forest - 2
lc_simpler[lc_simpler==40] = 2
lc_simpler[lc_simpler==50] = 2
lc_simpler[lc_simpler==60] = 2
lc_simpler[lc_simpler==61] = 2
lc_simpler[lc_simpler==62] = 2
lc_simpler[lc_simpler==70] = 2
lc_simpler[lc_simpler==71] = 2
lc_simpler[lc_simpler==72] = 2
lc_simpler[lc_simpler==80] = 2
lc_simpler[lc_simpler==81] = 2
lc_simpler[lc_simpler==82] = 2
lc_simpler[lc_simpler==90] = 2

# new code for shrubland - 3
lc_simpler[lc_simpler==100] = 3
lc_simpler[lc_simpler==120] = 3
lc_simpler[lc_simpler==121] = 3
lc_simpler[lc_simpler==122] = 3

# new code for herbaceous/grassland - 4
lc_simpler[lc_simpler==110] = 4
lc_simpler[lc_simpler==130] = 4

# new code for sparse areas - 5
lc_simpler[lc_simpler==140] = 5  #mosses/lichens
lc_simpler[lc_simpler==150] = 5
lc_simpler[lc_simpler==151] = 5
lc_simpler[lc_simpler==152] = 5
lc_simpler[lc_simpler==153] = 5

# new code for flooded/water areas - 6
lc_simpler[lc_simpler==160] = 6
lc_simpler[lc_simpler==170] = 6
lc_simpler[lc_simpler==180] = 6
lc_simpler[lc_simpler==210] = 6

# new code for bare areas - 7
lc_simpler[lc_simpler==200] = 7
lc_simpler[lc_simpler==201] = 7
lc_simpler[lc_simpler==202] = 7

# new code for urban/built up areas - 8
lc_simpler[lc_simpler==190] = 8

# new code for snow/ice areas - 9
lc_simpler[lc_simpler==220] = 9


class_rgb_mapping_simpler  = {
    # make colorings for 11,12,61,122,153,201
    
    0: (0, 0, 0),         #No Data 
    1: (255, 255, 100),   #Cropland
    12: (255, 200, 100),  #Cropland, trees
    2: (0, 100, 0),       #Tree cover
    3: (150, 100, 0),     #Shrubland
    4: (0, 150, 0),       #Grassland/herbaceous
    5: (255, 235, 175),   #Sparse vegetation 
    6: (0, 70, 200),      #Water/flooded bodies 
    7: (255, 245, 215),   #Bare areas  
    8: (195, 20, 0),      #Urban areas
    9: (255, 255, 255),   #Permanent snow and ice 
    
}

class_naming_simpler  = {
    # make colorings for 11,12,61,122,153,201
    
    0: "No Data ",
    1: "Cropland",
    12: "Cropland, trees",
    2: "Tree cover",
    3: "Shrubland",
    4: "Grassland/herbaceous",
    5:" Sparse vegetation" ,
    6: "Water/flooded ",
    7: "Bare areas"  ,
    8: "Urban/built-up areas",
    9: "Permanent snow and ice" ,
}


# visualize the simplified land cover
rgb_image_simpler  = np.zeros((lc_simpler.shape[1], lc_simpler.shape[2], 3), dtype=np.uint8)
for class_code, color in class_rgb_mapping_simpler.items():
    mask_simpler = lc_simpler[0,:,:] == class_code
    for i in range(3):  # Assign RGB channels
        rgb_image_simpler[:, :, i][mask_simpler] = color[i]

plt.imshow(rgb_image_simpler, origin='upper', aspect='equal')
plt.xticks([])
plt.yticks([])
#plt.savefig(f'land-cover-simpler-{year}-area-subset.{extent_string}.png', dpi=2000, bbox_inches="tight")
plt.show()


lccs_simpler_dataarray = xr.DataArray(
    data=lc_simpler,
    dims=ds.lccs_class.dims,
    coords=ds.lccs_class.coords,
    name="lccs_class_simpler"
)

ds["lccs_class_simpler"] = lccs_simpler_dataarray


#%% create xarray of aggregated majority land cover classes
latitudes = np.median(ds.lat_bounds.values, axis=1)
longitudes = np.median(ds.lon_bounds.values, axis=1)

factor = int(np.ceil(
    target_resolution / float(ds.attrs['geospatial_lat_resolution'])
    ))

def find_mean_of_groups(arr, group_size=None):
    if group_size==None:
        group_size=2
    # Reshape the array to group the elements into the desired size
    # This excludes leftover elements
    reshaped_arr = arr[:len(arr) - len(arr) % group_size].reshape(-1, group_size)
    # Calculate the mean of each group (along axis 1)
    mean_arr = np.mean(reshaped_arr, axis=1)
    return mean_arr

latitudes_agg = find_mean_of_groups(latitudes, group_size=factor)
longitudes_agg = find_mean_of_groups(longitudes, group_size=factor)

new_rows, new_cols = int(lc_simpler.shape[1]//factor), int(lc_simpler.shape[2]//factor)
reshaped = lc_simpler.reshape(new_rows, factor, new_cols, factor)
reshaped = np.transpose(reshaped, (0,2,1,3))
reshaped_flat = reshaped.reshape(reshaped.shape[0], reshaped.shape[1], factor*factor)

# maybeherea manual approach for computing mode using numpy might be
# more useful, especially for the case i want to make the aggregation
# not convert land to water, if water_num<factor*factor/2, but 
# the water code is the majority in the subset
print("Aggregating land cover through mode. This might take a while...")
lc_majority =  mode(reshaped_flat, axis=2).mode.squeeze()


# visualize the coarser, majority based aggregation, land cover
rgb_image_simpler_coarser  = np.zeros((lc_majority.shape[0], lc_majority.shape[1], 3), dtype=np.uint8)
for class_code, color in class_rgb_mapping_simpler.items():
    mask_simpler = lc_majority[:,:] == class_code
    for i in range(3):  # Assign RGB channels
        rgb_image_simpler_coarser[:, :, i][mask_simpler] = color[i]

plt.imshow(rgb_image_simpler_coarser, origin='upper', aspect='equal')
plt.xticks([])
plt.yticks([])
#plt.savefig(f'land-cover-simpler-coarser{target_resolution}-{year}-area-subset.{extent_string}.png', dpi=2000, bbox_inches="tight")
plt.show()


#%% extract the aggregated values to an .nc file
lc_coarse = xr.DataArray(
    lc_majority,
    dims=["latitude", "longitude"],
    coords={"latitude": latitudes_agg, "longitude": longitudes_agg},
    name="lc_coarse",
    attrs={
        "rgb_code":str(class_rgb_mapping_simpler),
        "class_name":str(class_naming_simpler),
        }
    )




lc_coarse_aligned = lc_coarse.interp(
    latitude=latitudes_subset,
    longitude=longitudes_subset,
    method="nearest"
    )


# visualize the coarser, majority based aggregation, 
# and also aligned and croped, land cover
lc_coarse_aligned_array = lc_coarse_aligned.values
rgb_image_simpler_coarser_aligned  = np.zeros(
    (lc_coarse_aligned_array.shape[0], lc_coarse_aligned_array.shape[1], 3), 
    dtype=np.uint8
    )
for class_code, color in class_rgb_mapping_simpler.items():
    mask_simpler = lc_coarse_aligned_array[:,:] == class_code
    for i in range(3):  # Assign RGB channels
        rgb_image_simpler_coarser_aligned[:, :, i][mask_simpler] = color[i]

plt.imshow(rgb_image_simpler_coarser_aligned, origin='upper', aspect='equal')
plt.xticks([])
plt.yticks([])
#plt.savefig(f'land-cover-simpler-coarser{target_resolution}-{year}-z-aligned.png', dpi=2000, bbox_inches="tight")
plt.show()

# make it also take the target extent as name
lc_coarse_aligned.to_netcdf(f"outputs\land-cover-{target_resolution}deg-year{year}.nc")

print("Done.")

'''
# this approach uses numpy to compute mode, and will be much
# faster once done, scipy.stats.mode is very slow for some reason...
def majority_numpy(x, axis=None):
    x = np.asarray(x)
    if axis is None:
        # Do not flatten, keep the data's original dimensions
        axis = 0  # Typically, you want to reduce over the last axis, adjust if needed
    values, counts = np.unique(x, return_counts=True, axis=axis)
    return values[np.argmax(counts)]

factor = 4
lc_coarse = ds.lccs_class_simpler.coarsen(lat=factor, lon=factor, boundary='trim').sum()
'''



