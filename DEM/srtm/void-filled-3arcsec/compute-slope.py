# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:27:28 2025

@author: yiann
"""
import os
import subprocess
import numpy as np
import xarray as xr
from osgeo import gdal
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colormaps


dem_file_name = "output.tif"  #must be tif format
scale_factor = 111120
#split DEM into strips, to compute slope in using appropriate
#scaling factor (which is latitude dependent)
strip_size = 0.1
export_nc_to_device = False
export_Gtiff_to_device = False
plot_morphography = False
#"delete" some datasets - might help with memory?
save_memory = True


#%%
cwd = Path.cwd()
input_dem = os.path.join(cwd, f"{dem_file_name}")
ds = gdal.Open(input_dem)

slope_file_name = f"{dem_file_name[:-4]}-slope.tif"
slope = os.path.join(cwd, slope_file_name)

gt = ds.GetGeoTransform()
rows, cols = ds.RasterYSize, ds.RasterXSize

latitudes = np.linspace(gt[3], gt[3] + gt[5] * rows, rows)
longitudes = np.linspace(gt[0], gt[0] + gt[1] * cols, cols)

dem = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

#find center latitude for scaling factor
top = gt[3]
bot = top + gt[5]*ds.RasterYSize

left = gt[0]
right = left + gt[1]*ds.RasterXSize

res_lat = abs(gt[5])
res_lon = abs(gt[1])


idx_rows = np.arange(rows)

center_latitudes = top - (idx_rows+0.5)*res_lat

# meters per degree for latitude-longitude
meters_latitude = scale_factor
meters_longitude = scale_factor * np.cos(np.radians(center_latitudes))

# spacing in meters for dataset latitude-longitude
meters_dx = res_lon * meters_longitude
meters_dy = res_lat * meters_latitude


#%% Slope computation - central differences
# Differences computation
dz_dx = np.zeros_like(dem, dtype=np.float32)
dz_dy = np.zeros_like(dem, dtype=np.float32)

# Compute dz/dx for each row using central differences:
for i in range(rows):
    # use the appropriate horizontal spacing
    spacing = meters_dx[i]
    # interior strips: central difference
    dz_dx[i, 1:-1] = (dem[i, 2:] - dem[i, :-2]) / (2 * spacing)
    # first and last strips use forward/backward differences.
    dz_dx[i, 0] = (dem[i, 1] - dem[i, 0]) / spacing
    dz_dx[i, -1] = (dem[i, -1] - dem[i, -2]) / spacing

# Compute dz/dy using central differences (vertical spacing is constant)
dz_dy[1:-1, :] = (dem[2:, :] - dem[:-2, :]) / (2 * meters_dy)
dz_dy[0, :] = (dem[1, :] - dem[0, :]) / meters_dy
dz_dy[-1, :] = (dem[-1, :] - dem[-2, :]) / meters_dy


# Compute gradient magnitude and slope
gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
slope_radians = np.arctan(gradient_magnitude)
slope_degrees = np.degrees(slope_radians)


#%% Exportations etc
if export_Gtiff_to_device == True:
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(slope_file_name, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(slope_degrees)
    out_band.SetNoDataValue(-9999)
    out_band.FlushCache()
    
    
if export_nc_to_device == True:
    dims = [
        "latitude",
        "longitude"
        ]
    
    ds_nc = xr.Dataset(
        {
            "dem" : (dims, dem),
           "slope" : (dims, slope_degrees),
        },
        coords = {
            "latitude" : latitudes,
            "longitude" : longitudes
            },
    )
    
    nc_filename_path = os.path.abspath(os.path.join(cwd, "..", "..", 
                                                    "outputs", "python")
                                       )
    nc_filename = os.path.join(nc_filename_path, 
                           f"{dem_file_name[:-4]}-morphography.nc"
                           )
    ds_nc.to_netcdf(nc_filename)


if save_memory == True:
    ds = None
    out_ds = None
    ds_nc = None


print(f"Slope calculation complete. Output saved as {slope_file_name}")

if plot_morphography == True:
    plt.title("Slope - in Degrees")
    plt.imshow(slope_degrees, cmap="magma")
    plt.savefig(f"{slope_file_name[:-4]}.png", dpi=1000)
    plt.show()


