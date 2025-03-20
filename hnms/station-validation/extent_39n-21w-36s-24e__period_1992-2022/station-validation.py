# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:02:29 2025

@author: yiann
"""

import math
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

stations = pd.read_csv("station-info.csv")
insitu = pd.read_csv("station-data.csv", index_col=0, parse_dates=True)

ld = xr.open_dataset("outputs_low_resolution_model.nc")
hd = xr.open_dataset("outputs_high_resolution_model.nc")


visualize = True
temporal_idx = 11



def get_bounding_square(target, resolution):
    """
    returns the latitude and longitude tuples of the
    specified target tuple in the IDW function
    
    should be fed to a temperature/value assignment
    function, to extract temperatures from the grids
    at those locations
    """
    x, y = target  # Latitude and Longitude
    
    lower_lat = np.floor(x / resolution) * resolution
    upper_lat = lower_lat + resolution
    
    lower_lon = np.floor(y / resolution) * resolution
    upper_lon = lower_lon + resolution
    
    lat_aligned = np.isclose(x, lower_lat)
    lon_aligned = np.isclose(y, lower_lon)
    
    # Case 1: The target is exactly on a grid point → Return that point
    if lat_aligned and lon_aligned:
        return [(x,y)]
    
    # Case 2: Only latitude is aligned → Return the left and right longitudes
    if lat_aligned:
        return [
            (x, lower_lon),  # Left point
            (x, upper_lon),  # Right point
        ]
    
    # Case 3: Only longitude is aligned → Return the above and below latitudes
    if lon_aligned:
        return [
            (lower_lat, y),  # Below point
            (upper_lat, y),  # Above point
        ]
    
    # Case 4: Neither latitude nor longitude is aligned → Return full bounding square
    return [
        (upper_lat, lower_lon),  # Upper left
        (upper_lat, upper_lon),  # Upper right
        (lower_lat, upper_lon),  # Lower right
        (lower_lat, lower_lon),  # Lower left
    ]

def get_known_points(target, dataarray, resolution):
    """
    Extracts temperature values from the dataset for the bounding square points
    and returns them in the format needed for IDW interpolation.

    Parameters:
        target (tuple): Target location as (latitude, longitude).
        dataset (xarray.Dataset): Xarray dataset containing temperature values.
        resolution (float): Grid resolution.

    Returns:
        list: Known points in the format [((lat, lon), value), ...]
    """
    bounding_points = get_bounding_square(target, resolution)
    known_points = []

    for lat, lon in bounding_points:
        # Extract the temperature value at this lat, lon
        temp_value = dataarray.sel(
            valid_time = dataarray.valid_time[temporal_idx],
            latitude=lat, 
            longitude=lon, 
            method="nearest"
            ).item()
        known_points.append(((lat, lon), temp_value))

    return known_points


def idw_interpolation__(target, known_points, power=2):
    """
    Calculates precipitation interpolated value at a target location using Inverse Distance Weighting (IDW).

    Parameters:
        target (tuple): Target location as (latitude, longitude).
        known_points (list of tuples): List of known points with ((lat, lon), value).
        power (int): Power parameter for IDW (default=2).

    Returns:
        float: Interpolated value at the target location.
    """
    if not known_points:
        print("No known points detected...")
        return None
    
    total_weight = 0
    weights = []
    interpolated_value = 0
    distances = []
    
    # Find distance of each known point from the target
    for (lat, lon), value in known_points:
        delta_lat = target[0] - lat
        delta_lon = target[1] - lon
        avg_lat = (target[0] + lat) / 2

        # Adjust longitude using cos(latitude)
        cos_lat = math.cos(math.radians(avg_lat))
        distance = math.sqrt(delta_lat ** 2 + (cos_lat * delta_lon) ** 2)
        distances.append(distance)
        
    # Perform the interpolation  
    for distance, value in zip(distances, known_points):
        # If value=nan, skip the weigting (value[1] is temp)
        if np.isnan(value[1]):
            continue
        weight = 1 / (distance ** power)
        weights.append(weight)
        total_weight += weight
        interpolated_value += weight * value[1]
        
        # Avoid division by zero if distance is very small
        if distance == 0:
            return value
    
    # Find the known point with minimum distance from the target
    # If it is nan, then make the target nan
    # I think this is better than quadrant identification
    min_distance_index = distances.index(min(distances))
    closest_point = known_points[min_distance_index]
    if np.isnan(closest_point[1]):
        interpolated_value = np.nan
    
    #total_weight = sum(weights)
    if total_weight == 0:
        return  np.nan
    
    return interpolated_value / total_weight


def idw_interpolation(target, dataarray, resolution, power=2):
    """
    Calculates precipitation interpolated value at a target location using Inverse Distance Weighting (IDW).

    Parameters:
        target (tuple): Target location as (latitude, longitude).
        known_points (list of tuples): List of known points with ((lat, lon), value).
        power (int): Power parameter for IDW (default=2).

    Returns:
        float: Interpolated value at the target location.
    """
    known_points = get_known_points(target, dataarray, resolution)
    if not known_points:
        print("No known points detected...")
        return None
    
    total_weight = 0
    weights = []
    interpolated_value = 0
    distances = []
    
    # Find distance of each known point from the target
    for (lat, lon), value in known_points:
        delta_lat = target[0] - lat
        delta_lon = target[1] - lon
        avg_lat = (target[0] + lat) / 2

        # Adjust longitude using cos(latitude)
        cos_lat = math.cos(math.radians(avg_lat))
        distance = math.sqrt(delta_lat ** 2 + (cos_lat * delta_lon) ** 2)
        distances.append(distance)
        
    # Perform the interpolation  
    for distance, value in zip(distances, known_points):
        # If value=nan, skip the weigting (value[1] is temp)
        if np.isnan(value[1]):
            continue
        weight = 1 / (distance ** power)
        weights.append(weight)
        total_weight += weight
        interpolated_value += weight * value[1]
        
        # Avoid division by zero if distance is very small
        if distance == 0:
            return value
    
    # Find the known point with minimum distance from the target
    # If it is nan, then make the target nan
    # I think this is better than quadrant identification
    min_distance_index = distances.index(min(distances))
    closest_point = known_points[min_distance_index]
    if np.isnan(closest_point[1]):
        interpolated_value = np.nan
    
    #total_weight = sum(weights)
    if total_weight == 0:
        print("Total weight is 0.")
        print("Check if target is outside dataarray bounds...")
        return  np.nan
    
    return interpolated_value / total_weight


#%% optikopoihsh
degree_spacing = 0.1
temporal_idx = temporal_idx

if visualize == True:
    
    t2m_pred = hd.t2mHD 
    t2m_res = hd.t2mHD + hd.resHD 
    t2mLD = ld.t2m 
    t2mLD_pred = ld.t2m_predLD
    times = hd.valid_time.values
    
    print('Mapping...')

    # order: t2mLD, t2mLD_pred, t2m_pred, t2m_res
    # dont change the order of the above
    # or change it everywhere the same below!
    titles = [
        (f"Low Resolution Predicted T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         ),
        (f"ERA5-Land T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         ),
        (f"Downscaled T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         ),
        (f"Residual Corrected Downscaled T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         )
        ]
    
    mins = [
        np.nanmin(t2mLD_pred.values[temporal_idx,:,:]),
        np.nanmin(t2mLD.values[temporal_idx,:,:]),
        np.nanmin(t2m_pred.values[temporal_idx,:,:]),
        np.nanmin(t2m_res.values[temporal_idx,:,:])
        ]
    maxes = [
        np.nanmax(t2mLD_pred.values[temporal_idx,:,:]),
        np.nanmax(t2mLD.values[temporal_idx,:,:]),
        np.nanmax(t2m_pred.values[temporal_idx,:,:]),
        np.nanmax(t2m_res.values[temporal_idx,:,:])
        ]
    
    fig, axes = plt.subplots(
        2, 2, 
        figsize=(12, 12), 
        subplot_kw={'projection': ccrs.PlateCarree()}
        )
    
    vmin = np.floor( np.min(mins) )
    vmax = np.ceil( np.max(maxes) )
    
    for ax, da, title in zip(
            axes.flat, 
            [t2mLD_pred,t2mLD,t2m_pred,t2m_res], 
            titles
            ):
        
        img = da.isel(valid_time=temporal_idx).plot.pcolormesh(
            ax=ax, cmap='inferno', transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, add_colorbar=False
            )

        ax.coastlines()
        ax.set_title(title)
        
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, linestyle="--", color="gray"
            )
        gl.xlocator = MultipleLocator(0.1)
        gl.ylocator = MultipleLocator(0.1)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.scatter(
            stations.lon, stations.lat, 
            c="k", marker="+", s=10,
            linewidth=0.8, alpha=0.75,
            zorder=10
            )
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.04, 0.78])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("Temperature [°C]")
        
    #plt.tight_layout()  
    #plt.savefig("multiplot.png", dpi=1000, bbox_inches="tight")
    plt.show() 