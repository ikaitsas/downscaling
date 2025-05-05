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
import ast
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
from collections import Counter


single_year = 2022
target_resolution = 0.1
target_extent = [41.8, 19.6, 35.8, 28.3] #N-W-S-E

export_to_device = True
visualize = False


#%% constants and functions
era5_resolution = 0.1  # resolution for era5-land
extent = [44, 17, 34, 32]
years = list(range(1992,2023))
reference_center = True
expand_to_all_directions = True


def find_mean_of_groups(arr, group_size=None):
    if group_size==None:
        group_size=2
    # Reshape the array to group the elements into the desired size
    # This excludes leftover elements
    reshaped_arr = arr[:len(arr) - len(arr) % group_size].reshape(-1, group_size)
    # Calculate the mean of each group (along axis 1)
    mean_arr = np.mean(reshaped_arr, axis=1)
    return mean_arr


def find_majority(arr, exception_value=6):  # 6 for water
    N, M, K = arr.shape
    result = np.empty((N, M), dtype=arr.dtype)
    
    for i in range(N):
        for j in range(M):
            values = arr[i, j, :]
            counts = Counter(values)
            sorted_by_frequency_counts = counts.most_common()

            top_value, top_count = sorted_by_frequency_counts[0]
            top_value = int(top_value)
            top_count = int(top_count)
            if top_value==exception_value & int((top_count/K))<=0.5:
                # If exception_value is the mode and there’s another option, use next best
                if len(sorted_by_frequency_counts)>1:
                    result[i, j] = sorted_by_frequency_counts[1][0]
                else:
                    result[i, j] = top_value  # fallback: all values are exception
            else:
                result[i, j] = top_value

    return result


def find_median_of_groups(arr, factor=2, height_axis=0, width_axis=1,
                          method="find_majority" #"scipy_mode" other option
                          ):
    # Right now method="scipy_mode" converts grid celsl where water (class=6)
    # is the plurality (majority but ratio<0.5) to water, which is not 
    # best for land cover, as land is more than water there
    # Please use method="find_majority" - it is still faster than scipy.mode...
    new_rows = int(arr.shape[height_axis]//factor) 
    new_cols = int(arr.shape[width_axis]//factor)
    
    reshaped = arr.reshape(new_rows, factor, new_cols, factor)
    reshaped = np.transpose(reshaped, (0,2,1,3))
    reshaped_flat = reshaped.reshape(
        reshaped.shape[0], reshaped.shape[1], factor*factor
        )
    
    if method == "find_majority":
        median_arr = find_majority(reshaped_flat, exception_value=6)
    elif method == "scipy_mode":
        mode(reshaped_flat, axis=2).mode.squeeze()
    else:
        median_arr = find_majority(reshaped_flat, exception_value=6)
    return median_arr


def extract_rgb(dataarray, rgb_attribute="rgb_code"):
    rgb_string = dataarray.attrs[rgb_attribute]
    rgb_dict = ast.literal_eval(rgb_string)
    return rgb_dict


def create_rgb_color_array(array, rgb_dict):
    # can only handle 3d and 2d arrays, with the 3d arrrays being
    # in the form (time, height, width), the 2d (height, width)
    if array.ndim == 3:
        rgb_image  = np.zeros(
            (array.shape[0], array.shape[1], array.shape[2], 3), 
            dtype=np.uint8
            )
        for code, color in rgb_dict.items():
            mask = array == code
            for i in range(3):  # Assign RGB channels
                rgb_image[:, :, :, i][mask] = color[i]
    elif array.ndim == 2:
        rgb_image  = np.zeros(
            (array.shape[0], array.shape[1], 3), 
            dtype=np.uint8
            )
        for code, color in rgb_dict.items():
            mask = array == code
            for i in range(3):  # Assign RGB channels
                rgb_image[:, :, i][mask] = color[i]
    else:
        raise ValueError("Input array must be 2D or 3D (time, height, width)")
    return rgb_image


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
if visualize == True:
    rgb_image = create_rgb_color_array(
        array=lc[0,:,:,], rgb_dict=class_color_mapping
        )
    
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
lc_simpler[lc_simpler==12] = 1  #tree crops might need different treatment
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

# new code for sparse areas - 5 / NOT PRACTICAL
lc_simpler[lc_simpler==140] = 4  #mosses/lichens
lc_simpler[lc_simpler==150] = 7
lc_simpler[lc_simpler==151] = 7
lc_simpler[lc_simpler==152] = 7
lc_simpler[lc_simpler==153] = 7

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
    4: (255, 165, 50),       #Grassland/herbaceous
    5: (255, 235, 175),   #Sparse vegetation 
    6: (0, 70, 200),      #Water/flooded bodies 
    7: (255, 245, 215),   #Bare areas  
    8: (195, 20, 0),      #Urban areas
    9: (255, 255, 255),   #Permanent snow and ice 
    
}

class_naming_simpler  = {
    # make colorings for 11,12,61,122,153,201
    
    0: "No Data ",
    1: "Agricultural",
    12: "Cropland, trees",
    2: "Tree cover",
    3: "Shrubland",
    4: "Grassland/herbaceous",
    5:" Sparse vegetation" ,
    6: "Water/flooded ",
    7: "Bare/sparse areas"  ,
    8: "Urban/built-up areas",
    9: "Permanent snow and ice" ,
}


# visualize the simplified land cover
if visualize == True:
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
factor = int(np.ceil(
    target_resolution / float(ds.attrs['geospatial_lat_resolution'])
    ))

print(f"Aggregating land cover to {target_resolution}deg. This might take a while...")
# make axis with size=1 detected and squeezed automatically/implicitly...
if lc_simpler.shape[0] == 1:
    lc_simpler = np.squeeze(lc_simpler, axis=0)
lc_majority = find_median_of_groups(arr=lc_simpler, factor=factor)


# visualize the coarser, majority based aggregation, land cover
if visualize == True:
    rgb_simpler_coarser = create_rgb_color_array(
        array=lc_majority, rgb_dict=class_rgb_mapping_simpler
        )
    
    plt.imshow(rgb_simpler_coarser, origin='upper', aspect='equal')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(f'land-cover-simpler-coarser{target_resolution}-{year}-area-subset.{extent_string}.png', dpi=2000, bbox_inches="tight")
    plt.show()


#%% extract the aggregated values to an .nc file
latitudes = np.median(ds.lat_bounds.values, axis=1)
longitudes = np.median(ds.lon_bounds.values, axis=1)

latitudes_agg = find_mean_of_groups(latitudes, group_size=factor)
longitudes_agg = find_mean_of_groups(longitudes, group_size=factor)

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
if visualize == True:
    lc_coarse_aligned_array = lc_coarse_aligned.values
    rgb_simpler_coarser_aligned = create_rgb_color_array(
        array=lc_coarse_aligned_array, rgb_dict=class_rgb_mapping_simpler
        )
    
    plt.imshow(rgb_simpler_coarser_aligned, origin='upper', aspect='equal')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(f'land-cover-simpler-coarser{target_resolution}-{year}-z-aligned.png', dpi=2000, bbox_inches="tight")
    plt.show()

# make it also take the target extent as name
if export_to_device == True:
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



