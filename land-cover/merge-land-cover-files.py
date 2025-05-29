# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:12:57 2025

@author: Giannis
"""

import os
import ast
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


years = list(range(1992,2023))
target_resolution = 0.01
target_extent = [41.8, 19.6, 35.8, 28.3] #N-W-S-E

file_paths = []

for year in years:
    file_path = f"outputs\land-cover-{target_resolution}deg-year{year}.nc"
    file_paths.append(file_path)

file_paths = sorted(file_paths)

dataarrays = [xr.open_dataarray(f) for f in file_paths]

combined = xr.concat(
    dataarrays, dim=xr.DataArray(years, dims="year", name="year")
    )

combined_file_path = (
    f"outputs\land-cover-{target_resolution}deg-"
    f"extent-{target_extent[0]}N.{target_extent[1]}W."
    f"{target_extent[2]}S.{target_extent[3]}E-"
    f"period{years[0]}-{years[-1]}.nc"
    )
combined.to_netcdf(combined_file_path)


#%% rgb handling functions
# these need to be imported into the land-cover.py script too...
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


rgb_code = extract_rgb(combined)
rgb_array = create_rgb_color_array(array=combined.values, rgb_dict=rgb_code)


#%% land cover visualization through the years
# tha to kanw pio omorfo me cartopy kapoia stigmh...
for i, year in enumerate(years):
    print(year)
    plt.imshow(rgb_array[i,:,:,:])
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Land Cover - {target_resolution}° Resolution - {year}")
    plt.show()
