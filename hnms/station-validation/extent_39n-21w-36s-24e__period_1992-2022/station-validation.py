# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:02:29 2025

@author: yiann
"""

import math
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

stations = pd.read_csv("station-info.csv")
insitu = pd.read_csv("station-data.csv", index_col=0, parse_dates=True)

ld = xr.open_dataset("outputs_low_resolution_model.nc")
hd = xr.open_dataset("outputs_high_resolution_model.nc")


visualize = False
temporal_idx = 11
target = (37.4067,22.7192)
dataarrayLD = ld.t2m
dataarrayHD = hd.t2mHD + hd.resHD


#%% functions
def find_bounding_box(target, dataarray):
    """
    The returned list of bounding points is shown clockwise
    if the target is right on top of a grid point, return target only
    if it is latitude/longitude aligned, return a bounding box of 6 points
    if it not aligned, return a 4 point square bounding box
    """
    target_lat, target_lon = target
    
    # Extract latitude and longitude arrays
    latitudes = dataarray.latitude.values
    longitudes = dataarray.longitude.values
    resolution = np.round(np.diff(longitudes)[0],5)

    # Handle cases where the target is outside bounds
    if (
        target[0] > latitudes.max() + resolution/2 or
        target[0] < latitudes.min() - resolution/2 or
        target[1] > longitudes.max() + resolution/2 or
        target[1] < longitudes.min() - resolution/2
    ):
        raise ValueError(
            "Target latitude or longitude is outside the dataset bounds."
            )
    
    # Find the two closest latitudes surrounding the target
    lower_lat = latitudes[latitudes <= target_lat].max() if \
    np.any(latitudes <= target_lat) else latitudes.min()
    upper_lat = latitudes[latitudes >= target_lat].min() \
    if np.any(latitudes >= target_lat) else latitudes.max()

    # Find the two closest longitudes surrounding the target
    lower_lon = longitudes[longitudes <= target_lon].max() if \
    np.any(longitudes <= target_lon) else longitudes.min()
    upper_lon = longitudes[longitudes >= target_lon].min() if \
    np.any(longitudes >= target_lon) else longitudes.max()
    
    lat_aligned = np.isclose(target_lat, upper_lat,rtol=1e-6) | \
    np.isclose(target_lat, lower_lat, rtol=1e-6)
    lon_aligned = np.isclose(target_lon, upper_lon, rtol=1e-6) | \
    np.isclose(target_lon, lower_lon, rtol=1e-6)
    
    if lat_aligned and lon_aligned:
        box = [(target_lat, target_lon)]
    
    elif lat_aligned :
        lat_idx = np.where(
            np.abs(latitudes-target_lat) == \
            np.min(np.abs(latitudes-target_lat))
            )[0][0]
        lat_above = latitudes[max(lat_idx - 1, 0)]
        lat_below = latitudes[min(lat_idx + 1, len(latitudes) - 1)]
        
        box = [
            (lat_above, lower_lon),  # Top-left
            (lat_above, upper_lon),  # Top-right
            (target_lat, upper_lon), # Center-right
            (lat_below, upper_lon),  # Bottom-right
            (lat_below, lower_lon),  # Bottom-left
            (target_lat, lower_lon)   # Center-left
        ]
    
    elif lon_aligned:
        lon_idx = np.where(
            np.abs(longitudes-target_lon) == \
            np.min(np.abs(longitudes-target_lon))
            )[0][0]
        lon_below = longitudes[max(lon_idx - 1, 0)]
        lon_above = longitudes[min(lon_idx + 1, len(longitudes) - 1)]

        box = [
            (upper_lat, lon_below),  # Top-left
            (upper_lat, target_lon),  # Top-center
            (upper_lat, lon_above), # Top-right
            (lower_lat, lon_above),  # Bottom-right
            (lower_lat, target_lon),  # Bottom-left
            (lower_lat, lon_below)   # Center-left
        ]
    
    else:
        # Standard case, with no alignment
        box = [
            (upper_lat, lower_lon),  # Top-left
            (upper_lat, upper_lon),  # Top-right
            (lower_lat, upper_lon),  # Bottom-right
            (lower_lat, lower_lon),  # Bottom-left
        ]
    
    box = list(dict.fromkeys(box))
    
    return box


def get_box_values(target, dataarray, timestamp):
    """
    Extracts temperature values from the dataset for the bounding square points
    and returns them in the format needed for IDW interpolation.

    Parameters:
        target (tuple): Target location as (latitude, longitude).
        dataset (xarray.Dataset): Xarray dataset containing temperature values.
        resolution (float): Grid resolution.
        timestamp (int): index of time dimension to perform the interpolation.

    Returns:
        list: Known points in the format [((lat, lon), value), ...]
    """
    bounding_points = find_bounding_box(target, dataarray)
    known_points = []

    for lat, lon in bounding_points:
        # Extract the temperature value at this lat, lon
        temp_value = dataarray.sel(
            valid_time = dataarray.valid_time[timestamp],
            latitude=lat, 
            longitude=lon, 
            method="nearest"
            ).item()
        known_points.append(((lat, lon), temp_value))

    return known_points


def idw_interpolation(target, dataarray, timestamp, power=2):
    """
    Calculates interpolated value at a target location using 
    Inverse Distance Weighting (IDW).

    Parameters:
        target (tuple): Target location as (latitude, longitude).
        known_points (list of tuples): List of known points 
        with ((lat, lon), value).
        power (int): Power parameter for IDW (default=2).

    Returns:
        float: Interpolated value at the target location.
    """
    resolution = np.round(np.diff(dataarray.longitude.values)[0],5)
    # Check if target lies outside of dataarray bounds
    if (
        target[0] > dataarray.latitude.values.max()+resolution/2 or
        target[0] < dataarray.latitude.values.min()-resolution/2 or
        target[1] > dataarray.longitude.values.max()+resolution/2 or
        target[1] < dataarray.longitude.values.min()-resolution/2
    ):
        print("Target is out of Dataarray Bounds.")
        return None
    
    known_points = get_box_values(target, dataarray, timestamp)
    if not known_points:
        print("No known points detected...")
        return None
    
    total_weight = 0
    weights = []
    interpolated_value = 0
    distances = []
    values = []
    
    # Find distance of each known point from the target
    for (lat, lon), value in known_points:
        delta_lat = target[0] - lat
        delta_lon = target[1] - lon
        avg_lat = (target[0] + lat) / 2

        # Adjust longitude using cos(latitude)
        cos_lat = math.cos(math.radians(avg_lat))
        distance = math.sqrt(delta_lat ** 2 + (cos_lat * delta_lon) ** 2)
        distances.append(distance)
        values.append(value)
        
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


def idw_interpolation_across_time(target, dataarray, power=2):
    """
    Applies Inverse Distance Weighting (IDW) interpolation for a target 
    location across the entire valid_time axis of the dataarray.

    Parameters:
        target (tuple): Target location as (latitude, longitude).
        dataarray (xarray.DataArray): Data array with valid_time, 
        latitude, and longitude dimensions.
        resolution (float): Resolution of known points selection.
        power (int): Power parameter for IDW (default=2).

    Returns:
        xarray.DataArray: Interpolated values across valid_time.
    """
    resolution = np.round(np.diff(dataarray.longitude.values)[0],5)

    # Check if target lies outside of dataarray bounds
    if (
        target[0] > dataarray.latitude.values.max()+resolution/2 or
        target[0] < dataarray.latitude.values.min()-resolution/2 or
        target[1] > dataarray.longitude.values.max()+resolution/2 or
        target[1] < dataarray.longitude.values.min()-resolution/2
    ):
        print("Target is out of Dataarray Bounds.")
        return None
    
    interpolated_values = np.full((len(dataarray.valid_time), 1, 1), np.nan)  
    
    for timestamp in range(len(dataarray.valid_time)):
        known_points = get_box_values(target, dataarray, timestamp)
        if not known_points:
            interpolated_values.append(np.nan)
            continue
        
        total_weight = 0
        interpolated_value = 0
        distances = []
        values = []
        
        # Compute distances
        for (lat, lon), value in known_points:
            delta_lat = target[0] - lat
            delta_lon = target[1] - lon
            avg_lat = (target[0] + lat) / 2
            cos_lat = math.cos(math.radians(avg_lat))
            distance = math.sqrt(delta_lat ** 2 + (cos_lat * delta_lon) ** 2)
            distances.append(distance)
            values.append(value)
        
        # Perform the interpolation  
        for distance, value in zip(distances, known_points):
            if np.isnan(value[1]):
                continue
            weight = 1 / (distance ** power)
            total_weight += weight
            interpolated_value += weight * value[1]
            
            if distance == 0:
                interpolated_values.append(value[1])
                break
        else:
            if total_weight != 0:
                interpolated_values[timestamp, 0, 0] = \
                interpolated_value / total_weight
        
    # Find the known point with minimum distance from the target
    # If it is nan, then make the target nan
    # I think this is better than quadrant identification
    min_distance_index = distances.index(min(distances))
    closest_point = known_points[min_distance_index]
    if np.isnan(closest_point[1]):
        interpolated_values = np.full(
            (len(dataarray.valid_time), 1, 1), 
            np.nan
            )
    
    dataarray_name = dataarray.name
    if dataarray_name is None:
        dataarray_name = 't2m'
            
    return xr.DataArray(
        interpolated_values, 
        coords={
            "valid_time": dataarray.valid_time, 
            "latitude": [target[0]], 
            "longitude": [target[1]]
        }, 
        dims=["valid_time", "latitude", "longitude"],
        name=dataarray_name
    )


def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)

def unbiased_root_mean_squared_error(y_true, y_pred):
    mbe = mean_bias_error(y_true, y_pred)
    ubmse = np.mean( (y_pred - y_true - mbe)**2 )
    return np.sqrt( ubmse )

def unbiased_root_mean_squared_error2(y_true, y_pred):
    mbe = mean_bias_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return np.sqrt( rmse**2 - mbe**2 )

def compute_metrics(group, mask):
    #mask = group['downscaled'].notna()  # Ensure valid comparisons
    #group = group.drop(columns=["month"], errors="ignore")
    #SHOULD PROBABLY REMOVE MASK PARAMETER
    masked_group = group[mask.loc[group.index]]
    
    return pd.Series({
        'mseLD': mean_squared_error(
            masked_group['insitu'], masked_group['idw']
            ),
        'mseHD': mean_squared_error(
            masked_group['insitu'], masked_group['downscaled']
            ),
        'maeLD': mean_absolute_error(
            masked_group['insitu'], masked_group['idw']
            ),
        'maeHD': mean_absolute_error(
            masked_group['insitu'], masked_group['downscaled']
            ),
        'mbeLD': mean_bias_error(
            masked_group['insitu'], masked_group['idw']
            ),
        'mbeHD': mean_bias_error(
            masked_group['insitu'], masked_group['downscaled']
            ),
        #'r2LD': r2_score(
        #    group.loc[mask, 'insitu'], group.loc[mask, 'idw']
        #    ),
        #'r2HD': r2_score(
        #    group.loc[mask, 'insitu'], group.loc[mask, 'downscaled']
        #    )
        'ubrmseLD': unbiased_root_mean_squared_error(
            masked_group['insitu'], masked_group['idw']
            ),
        'ubrmseHD': unbiased_root_mean_squared_error(
            masked_group['insitu'], masked_group['downscaled']
            ),
        'mapeLD': mean_absolute_percentage_error(
            masked_group['insitu'], masked_group['idw']
            ),
        'mapeHD': mean_absolute_percentage_error(
            masked_group['insitu'], masked_group['downscaled']
            ),
    })


#%% dokimastiko
temp_target = idw_interpolation(
        target=target, dataarray=ld.t2m, 
        timestamp=temporal_idx,
        power=2
        )
print(f'Temperature at target: {temp_target:.2f}')

interp = idw_interpolation_across_time(
        target=target, dataarray=ld.t2m ,power=2
        )
for time,value in zip(interp.valid_time.values,interp.values):
    print(time,np.round(value[0][0],2))


#%% optikopoihsh
degree_spacing = 0.1
temporal_idx = temporal_idx

t2m_pred = hd.t2mHD 
t2m_res = hd.t2mHD + hd.resHD 
t2mLD = ld.t2m 
t2mLD_pred = ld.t2m_predLD
times = hd.valid_time.values

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

if visualize == True:
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
            c="darkslategrey", marker="+", s=12,
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


#%% paidikh xara
dfLD = insitu.loc[insitu.index.isin(hd.valid_time.values), :]
dfHD = insitu.loc[insitu.index.isin(hd.valid_time.values), :]
dfSITE = insitu.loc[insitu.index.isin(hd.valid_time.values), :]

'''
Pithanotata na yparxei kai grhgoroteros tropos
Alla autos douleuei sigoura kala
Prosoxh sta indexes, exei mismatch logw multiindex
sta dataarray derived products
me kamia lambda function mhpws??
'''
print("Performing IDW interpolation on LD & HD Data...")
for i in range(len(stations)):
    station = stations.iloc[i]
    
    station_code = station.iloc[1]
    station_name = station.iloc[2]
    
    station_location = ( station.iloc[3], station.iloc[4] )
    
    print(f"Station: {station_name} ({station_code})")
    
    seriesLD = idw_interpolation_across_time(
        target=station_location, dataarray=dataarrayLD
        ).to_series()
    
    seriesHD = idw_interpolation_across_time(
        target=station_location, dataarray=dataarrayHD
        ).to_series()
    
    seriesSITE = insitu.loc[
        insitu.index.isin(hd.valid_time.values), str(station_code)
        ]
    
    if visualize == True:
        plt.plot(hd.valid_time.values, seriesLD, linestyle="--")
        plt.plot(hd.valid_time.values, seriesSITE, c="r", alpha=0.75)
        plt.plot(hd.valid_time.values, seriesHD, c="k")
        plt.title(f'2m Temperature - {station_name} ({station_code})')
        plt.legend(["IDW", "In-situ", "Downscaled"], prop={'size': 7}, framealpha=0.3)
        plt.xticks(rotation=30)
        plt.ylabel("Temperature  [°C]")
        plt.grid()
        #plt.savefig(f'timeseries-{station_code}-{station_name}.png', dpi=300)
        plt.show()
    
    dfLD.loc[:,str(station_code)] = seriesLD.values
    dfHD.loc[:,str(station_code)] = seriesHD.values


#%% Performance metrics
# These below get stacked "horizontally" (per station first)
# if future_stack=True specified, dont specify dropna=False
dfLD_stacked = dfLD.stack(future_stack=True)
dfHD_stacked = dfHD.stack(future_stack=True)
dfSITE_stacked = dfSITE.stack(future_stack=True)


if visualize == True:
    x= np.linspace(3, 32, 100)
    plt.grid()
    plt.scatter(dfSITE_stacked, dfLD_stacked, s=3, alpha=0.95)
    plt.scatter(dfSITE_stacked, dfHD_stacked, s=3, alpha=0.75)
    plt.ylabel("Modeled Temperature  [°C]")
    plt.xlabel("Insitu Temperature  [°C]")
    plt.legend(["IDW", "Downscaled"])
    plt.plot(x, x, c="k", linestyle="--", alpha=0.75)
    plt.title("Temperature Scatter Plot")
    plt.axis("square")
    #plt.savefig("scatter-plot-monthly.png", dpi=1000)
    plt.show()


df_stacked = pd.concat([dfLD_stacked, dfHD_stacked, dfSITE_stacked], axis=1)
df_stacked.columns = ["idw", "downscaled", "insitu"]
df_stacked['month'] = df_stacked.index.get_level_values(0).month
df_stacked['station_code'] = df_stacked.index.get_level_values(1).astype(int)

mask = ~np.isnan(dfSITE_stacked) & ~np.isnan(dfLD_stacked)
df_stacked_mask = df_stacked.loc[mask,:]


mseLD = mean_squared_error(df_stacked_mask.insitu, df_stacked_mask.idw)
mseHD = mean_squared_error(df_stacked_mask.insitu, df_stacked_mask.downscaled)
maeLD = mean_absolute_error(df_stacked_mask.insitu, df_stacked_mask.idw)
maeHD = mean_absolute_error(df_stacked_mask.insitu, df_stacked_mask.downscaled)
r2LD = r2_score(df_stacked_mask.insitu, df_stacked_mask.idw)
r2HD = r2_score(df_stacked_mask.insitu, df_stacked_mask.downscaled)

ubrmseLD = unbiased_root_mean_squared_error(
    df_stacked_mask.insitu, df_stacked_mask.idw)
ubrmseHD = unbiased_root_mean_squared_error(
    df_stacked_mask.insitu, df_stacked_mask.downscaled)

ubrmseLD2 = unbiased_root_mean_squared_error2(
    df_stacked_mask.insitu, df_stacked_mask.idw)
ubrmseHD2 = unbiased_root_mean_squared_error2(
    df_stacked_mask.insitu, df_stacked_mask.downscaled)

mapeLD = mean_absolute_percentage_error(
    df_stacked_mask.insitu, df_stacked_mask.idw
    )
mapeHD = mean_absolute_percentage_error(
    df_stacked_mask.insitu, dfHD_stacked[mask]
    )

mbeLD = np.mean(df_stacked_mask.idw - df_stacked_mask.insitu)
mbeHD = np.mean(df_stacked_mask.downscaled - df_stacked_mask.insitu)


month_group_key = "month"
monthly_metrics = df_stacked[mask].groupby([month_group_key])[
    ["idw","downscaled","insitu"]
    ].apply(
    lambda group: compute_metrics(group, mask)
    )
monthly_metrics["idw_variance"] = df_stacked[mask].groupby(
    month_group_key)[["idw"]].var()
monthly_metrics["downscaled_variance"] = df_stacked[mask].groupby(
    month_group_key)[["downscaled"]].var()
monthly_metrics["insitu_variance"] = df_stacked[mask].groupby(
    month_group_key)[["insitu"]].var()


station_group_key = "station_code"
station_metrics = df_stacked[mask].groupby([station_group_key])[
    ["idw","downscaled","insitu"]
    ].apply(
    lambda group: compute_metrics(group, mask)
    )
station_metrics["idw_variance"] = df_stacked[mask].groupby(
    station_group_key)[["idw"]].var()
station_metrics["downscaled_variance"] = df_stacked[mask].groupby(
    station_group_key)[["downscaled"]].var()
station_metrics["insitu_variance"] = df_stacked[mask].groupby(
    station_group_key)[["insitu"]].var()


#%% Metrics Visualization
if visualize == True:
    for metric in ["mse", "mae", "mbe", "ubrmse"]:
        print(metric+"LD", metric+"HD")
        # Merge the 2 plots in a subplot...
        # stations.name[stations.WMO_code.isin(station_metrics.index)]
        
        plt.plot(range(len(station_metrics)), station_metrics[metric+"LD"])
        plt.plot(range(len(station_metrics)), station_metrics[metric+"HD"])
        plt.xticks(
            range(len(station_metrics)), station_metrics.index,
            rotation=35
            )
        plt.grid()
        plt.title(f"{metric.upper()} - Per Station")
        plt.legend(["IDW", "Downscaled"])
        plt.xlabel("Station Code")
        
        if metric == "mse":
            ylabel = "Error [°C²]"
        else:
            ylabel = "Error [°C]"
        plt.ylabel(ylabel)
        
        if metric == "mbe":
            plt.ylim(top=0)
        
        plt.show()
        
        
        plt.plot(range(len(monthly_metrics)), monthly_metrics[metric+"LD"])
        plt.plot(range(len(monthly_metrics)), monthly_metrics[metric+"HD"])
        plt.xticks(
            range(len(monthly_metrics)), monthly_metrics.index,
            rotation=35
            )
        plt.grid()
        plt.title(f"{metric.upper()} - Per Month")
        plt.legend(["IDW", "Downscaled"])
        plt.xlabel("Month")
        
        if metric == "mse":
            ylabel = "Error [°C²]"
        else:
            ylabel = "Error [°C]"
        plt.ylabel(ylabel)
        
        if metric == "mbe":
            plt.ylim(top=0)
        
        plt.show()


#%% koments
'''
correlation = df_stacked[["insitu", "idw", "downscaled"]].corr()
correlation_each_month = df_stacked.groupby("month")[["insitu", "idw", "downscaled"]].corr()
correlation_per_station = df_stacked.groupby("station_code")[["insitu", "idw", "downscaled"]].corr()

df_stacked["residuals_idw"] = df_stacked["insitu"] - df_stacked["idw"]
df_stacked["residuals_downscaled"] = df_stacked["insitu"] - df_stacked["downscaled"]

residuals_variance = df_stacked.groupby("month")[["residuals_idw", "residuals_downscaled"]].var()
residuals_variance["insitu"] = df_stacked.groupby("month")[["insitu"]].var()
'''

'''
mseLD_months, mseHD_months = [], []
maeLD_months, maeHD_months = [], []
# Per month r2 producing VERY STRANGE values
# This might have to do with variance reduction at monthly scales??
# WHole year: var = 51.20 for insitu
# Per month: 1.48 < var <4.27 for insitu
# Maybe will ad seasonal aggregation, there r2 might not be dogshit
for month in range(1,13):
    per_month = df_stacked[df_stacked.month==month]
    
    mseLD_ = mean_squared_error(
        per_month.insitu[mask], per_month.idw[mask]
        )
    mseHD_ = mean_squared_error(
        per_month.insitu[mask], per_month.downscaled[mask]
        )
    mseLD_months.append(mseLD_)
    mseHD_months.append(mseHD_)
    
    maeLD_ = mean_absolute_error(
        per_month.insitu[mask], per_month.idw[mask]
        )
    maeHD_ = mean_absolute_error(
        per_month.insitu[mask], per_month.downscaled[mask]
        )
    maeLD_months.append(maeLD_)
    maeHD_months.append(maeHD_)
'''

'''
mseLD_stations, mseHD_stations = [], []
maeLD_stations, maeHD_stations = [], []
for station_code in np.unique(df_stacked[mask].index.get_level_values(1).astype(int)):
    per_station = df_stacked[df_stacked.station_code==station_code]
    
    mseLD_ = mean_squared_error(
        per_station.insitu[mask], per_station.idw[mask]
        )
    mseHD_ = mean_squared_error(
        per_station.insitu[mask], per_station.downscaled[mask]
        )
    mseLD_stations.append(mseLD_)
    mseHD_stations.append(mseHD_)
    
    maeLD_ = mean_absolute_error(
        per_station.insitu[mask], per_station.idw[mask]
        )
    maeHD_ = mean_absolute_error(
        per_station.insitu[mask], per_station.downscaled[mask]
        )
    maeLD_stations.append(maeLD_)
    maeHD_stations.append(maeHD_)
'''
    
'''
fig, ax = plt.subplots(
    figsize=(7, 7), subplot_kw={"projection": ccrs.PlateCarree()}
    )

img = hd.dem.plot.pcolormesh(
    ax=ax, cmap='inferno_r', transform=ccrs.PlateCarree(),
    vmin=0, vmax=hd.dem.values.max(), add_colorbar=False
    )

ax.coastlines()
ax.set_title("Elevation - 1km Resolution")

gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, linestyle="--", color="gray"
    )
gl.xlocator = MultipleLocator(0.1)
gl.ylocator = MultipleLocator(0.1)
gl.top_labels = False
gl.right_labels = False

ax.scatter(
    stations.lon, stations.lat, 
    c="g", marker="+", s=10,
    linewidth=0.8, alpha=0.75,
    zorder=10
    )

#fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.11, 0.04, 0.78])
cbar = fig.colorbar(img, cax=cbar_ax)
cbar.set_label("Elevation [m]")
    
#plt.tight_layout()  
#plt.savefig("elevation.png", dpi=1000, bbox_inches="tight")
plt.show() 
'''