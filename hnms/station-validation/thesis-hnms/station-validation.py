# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:02:29 2025

@author: yiann
"""

import sys
from typing import Tuple

import math
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import probplot
from scipy.stats import linregress
from scipy.special import kl_div
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy.stats import anderson_ksamp

from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import skew

from arch.bootstrap import optimal_block_length
from arch.bootstrap import MovingBlockBootstrap
from arch.bootstrap import StationaryBootstrap

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator


coarse_resolution = 0.1
target_resolution = 0.05

stations = pd.read_csv(
    "Valid_HNMS_Stations_Info__N41.8-W19.6-S35.8-E28.3__Period1992-2022.csv"
    )
insitu = pd.read_csv("TG-m__N41.8-W19.6-S35.8-E28.3__Period1992-2022.csv", 
                     index_col=0, parse_dates=True
                     )

ld = xr.open_dataset(f"outputs-{coarse_resolution}deg-with-lc.nc")
hd = xr.open_dataset(f"outputs-{target_resolution}deg-with-lc.nc")


visualize = False
save_figures = False

temporal_idx = 47
target = (37.4067,22.7192)
dataarrayLD = ld.t2m
dataarrayHD = hd.t2mHD + hd.resHD


#log_file = open(f'printed_output{target_resolution}.log', 'w', encoding="utf-8")
#sys.stdout = log_file


#%% functions for idw
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
            if distance == 0:
                interpolated_values[timestamp, 0, 0] = value[1]
                total_weight = 1
                break
            if np.isnan(value[1]):
                continue
            weight = 1 / (distance ** power)
            total_weight += weight
            interpolated_value += weight * value[1]
            
            
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


#%% metrics and statistics functions
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
        'rmseLD': root_mean_squared_error(
            masked_group['insitu'], masked_group['idw']
            ),
        'rmseHD': root_mean_squared_error(
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

def ecdf(data):
    '''
    Returns an array of the sorted data
    And the ecdf value for each sorted datapoint
    '''
    return np.sort(data), np.arange(1, len(data) + 1) / len(data)


def yule_kendall_skewness(data):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)

    if q3 - q1 == 0:
        return np.nan  # Avoid division by zero if data is constant
    return (q3 + q1 - 2 * q2) / (q3 - q1)


#%% rounding functions for scatter plot
def round_sigfig(x):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))))

def round_value_to_match_std(value, std_rounded):
    # Convert to string, split by decimal
    s = f"{std_rounded:.10f}".rstrip('0')  # string without trailing zeros
    if '.' in s:
        decimal_places = len(s.split('.')[1])
    else:
        decimal_places = 0
    return round(value, decimal_places)

def make_legend_label(slope, slope_std, intercept, intercept_std):
    # Round stds to 1 significant digit
    slope_std_rounded = round_sigfig(slope_std)
    intercept_std_rounded = round_sigfig(intercept_std)

    # Round slope/intercept to same decimal place as stds
    slope_rounded = round_value_to_match_std(slope, slope_std_rounded)
    intercept_rounded = round_value_to_match_std(intercept, intercept_std_rounded)

    # Format the string nicely, using ± symbol
    label = (f"y = ({slope_rounded} ± {slope_std_rounded})x "
             f"+ ({intercept_rounded} ± {intercept_std_rounded})")
    return label


#%% bootstrap testing functions
def compute_optimal_block_length1(series):
    n = len(series)
    acf_values = acf(series, nlags=n-1, fft=True)
    
    # Compute the "correlation length"
    k = np.argmax(np.abs(acf_values) < np.exp(-1)) # First lag where ACF < 1/e
    optimal_block = int(np.ceil(n**(1/3) * k**(2/3)))
    
    return max(2, min(optimal_block, n//2))  # Keep within reasonable bounds

    
def multi_seasonal_block_length(data, candidate_periods=[3, 6, 12, 24, 36, 48, 60]):
    max_period_tolerated = len(data)//2
    candidate_periods = [period for period in candidate_periods if period <= max_period_tolerated]
    
    acf_vals = acf(data, nlags=max(candidate_periods))
    strengths = [acf_vals[lag] for lag in candidate_periods]
    dominant_period = candidate_periods[np.argmax(np.abs(strengths))]
    
    # Ensure block length isn't too large
    max_reasonable = int(len(data)**(1/2))
    return min(dominant_period, max_reasonable)


###### PETAMAAAA
def compute_vif_significant_lags(series, cutoff=None, nlags=None, alpha=0.05):
    n = len(series)
    acf_vals, confint = acf(series, nlags=nlags, alpha=alpha)
    
    # for below, might be simpler to emulate AR(1) or AR(2) models, and
    # only keep these autoregression lags
    # for monthly temperature in particular, AR(2) is better? (Wilks, 1997)
    # they are selected with cutoff == 1 or 2
    
    V = 1  # Start with base
    if cutoff == 1:
        significant_vals = acf_vals[1:cutoff+1]
        for k,rho_k in enumerate(significant_vals):
            V += 2 * (1 - k/n) * rho_k  # Add only 1 lag
    elif cutoff == 2:
        significant_vals = acf_vals[1:cutoff+1]
        for k,rho_k in enumerate(significant_vals):  
            V += 2 * (1 - k/n) * rho_k  # Add only 2 lags
    else:
        # corresponds to the shaded area in plot_acf
        confint_centered = np.abs(confint[:,1]-acf_vals) # auto kaloo
        significant_lags_idx = np.where(np.abs(acf_vals)>confint_centered)[0][1:]
        significant_vals = acf_vals[significant_lags_idx]
        for k,rho_k in enumerate(significant_vals):  # exclude lag 0
            V += 2 * (1 - k/n) * rho_k  # Add only significant lags
    return V


def bootstrap_test_autocorrelated(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    method: str = "moving_block",
    n_bootstrap: int = 10000
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compare two autocorrelated time series using moving block bootstrap.
    
    Args:
        x: First time series array
        y: Second time series array
        block_size: Block length for bootstrap (should capture autocorrelation)
        ci_percent: Desired confidence interval percentage (e.g., 95 for 95% CI)
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Tuple containing:
        - observed_mean_diff: The observed mean difference (x.mean() - y.mean())
        - ci: Confidence interval bounds as (lower, upper)
        - p_value: Two-sided p-value for the mean difference
    """
    # Validate inputs
    assert len(x) == len(y), "Arrays must be of equal length"
    assert block_size > 0, "Block size must be positive"
    
    # Calculate observed statistic
    observed_mean_diff = np.mean(x) - np.mean(y)
    
    # Combine data for paired bootstrap
    combined = np.column_stack((x, y))
    
    # Initialize bootstrap
    if method == "moving_block":
        bs = MovingBlockBootstrap(block_size, combined, seed=1234)
    else:
        bs = StationaryBootstrap(block_size, combined, seed=1234)
    
    # Run bootstrap
    bootstrap_diffs = []
    resampled_shapes = []
    for result in bs.bootstrap(n_bootstrap):
        resampled_data = result[0][0]  # Get resampled array
        x_resampled = resampled_data[:, 0]
        y_resampled = resampled_data[:, 1]
        bootstrap_diff = np.mean(x_resampled) - np.mean(y_resampled)
        # subtracting observed_mean_diff, not S0 (0 zero)
        # thisis done to reflect the null hypothesis
        bootstrap_diff = bootstrap_diff - observed_mean_diff
        bootstrap_diffs.append(bootstrap_diff)
        
        resampled_shapes.append(resampled_data.shape)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate p-value (two-sided)
    # Center the bootstrap distribution at 0 for null hypothesis
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_mean_diff))
    
    return observed_mean_diff, p_value

def bootstrap_test_autocorrelated2(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    method: str = "moving_block",
    n_bootstrap: int = 10000
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compare two autocorrelated time series using moving block bootstrap.
    
    Args:
        x: First time series array
        y: Second time series array
        block_size: Block length for bootstrap (should capture autocorrelation)
        ci_percent: Desired confidence interval percentage (e.g., 95 for 95% CI)
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Tuple containing:
        - observed_mean_diff: The observed mean difference (x.mean() - y.mean())
        - ci: Confidence interval bounds as (lower, upper)
        - p_value: Two-sided p-value for the mean difference
    """
    # Validate inputs
    assert len(x) == len(y), "Arrays must be of equal length"
    assert block_size > 0, "Block size must be positive"
    
    # Calculate observed statistic
    observed_mean_diff = np.mean(x) - np.mean(y)
    
    # Calculate centered paired differences to enforce null hypothesis
    d = x - y
    d_centered = d - np.mean(d)

    # Construct null sample where difference mean = 0
    x_null = y + d_centered  # so mean(x_null) == mean(y)
    
    # Combine data for paired bootstrap
    combined = np.column_stack((x_null, y))
    
    # Initialize bootstrap
    if method == "moving_block":
        bs = MovingBlockBootstrap(block_size, combined, seed=1234)
    else:
        bs = StationaryBootstrap(block_size, combined, seed=1234)
    
    # Run bootstrap
    bootstrap_diffs = []
    for result in bs.bootstrap(n_bootstrap):
        resampled_data = result[0][0]  # Get resampled array
        x_resampled = resampled_data[:, 0]
        y_resampled = resampled_data[:, 1]
        bootstrap_diff = np.mean(x_resampled) - np.mean(y_resampled)
        
        bootstrap_diffs.append(bootstrap_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate p-value (two-sided)
    # Center the bootstrap distribution at 0 for null hypothesis
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_mean_diff))
    
    return observed_mean_diff, p_value

def bootstrap_test_autocorrelated3(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    bootstrap_method: str = "moving_block",
    n_bootstrap: int = 10000,
) -> Tuple[float, float]:
    """
    Compare two autocorrelated time series using block bootstrap.
    Returns: (observed_mean_diff, (ci_lower, ci_upper), p_value)
    """
    # Calculate the pooled mean (assuming H₀: μₓ = μᵧ is true)
    pooled_mean = (np.mean(x) + np.mean(y)) / 2
    
    # Center both series around the pooled mean (enforcing H₀)
    x_centered = x - np.mean(x) + pooled_mean
    y_centered = y - np.mean(y) + pooled_mean
    
    combined = np.column_stack((x_centered, y_centered))
    
    # Validate inputs and initialize bootstrap
    if bootstrap_method == "moving_block":
        bs = MovingBlockBootstrap(block_size, combined)
    else:
        bs = StationaryBootstrap(block_size, combined)

    # Observed statistic
    observed_mean_diff = np.mean(x) - np.mean(y)

    # Bootstrap the mean difference
    bootstrap_diffs = []
    for result in bs.bootstrap(n_bootstrap):
        resampled_data = result[0][0]
        x_resampled = resampled_data[:, 0]
        y_resampled = resampled_data[:, 1]
        bootstrap_diffs.append(np.mean(x_resampled) - np.mean(y_resampled))
    
    bootstrap_diffs = np.array(bootstrap_diffs)

    # Correct p-value calculation: Fraction of bootstrap diffs MORE EXTREME than observed
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_mean_diff))

    return observed_mean_diff, p_value


def permutation_test(diff, n_permutations=10000, alternative='greater', seed=None):
    """
    Permutation test for paired differences.

    Parameters:
    - diff: array-like, differences (e.g., error1 - error2)
    - n_permutations: number of permutations (default: 10,000)
    - alternative: 'greater', 'less', or 'two-sided'
    - seed: optional int, for reproducibility

    Returns:
    - p-value (float)
    """
    if seed is not None:
        np.random.seed(seed)

    diff = np.array(diff)
    diff = diff[diff != 0]  # remove ties
    
    nonzero_diff = diff[diff != 0]  # Exclude ties (zero differences)
    if len(nonzero_diff) == 0:
        print("⚠️ All differences are zero — no variation to test. Returning p = 1.0.")
        return 1.0

    observed_stat = np.sum(diff)
    perm_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        signs = np.random.choice([1, -1], size=len(diff))
        perm_stats[i] = np.sum(diff * signs)

    if alternative == 'greater':
        p_value = np.mean(perm_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(perm_stats <= observed_stat)
    elif alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_stats) >= abs(observed_stat))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return p_value

def block_permutation_test(diff, block_size=12, n_permutations=10000, alternative='greater', random_state=None):
    """
    Perform a block permutation test on paired difference data with temporal dependence.
    
    Parameters:
        diff (np.array): Array of paired differences (e.g., error1 - error2), shape (n,).
        block_size (int): Length of each block (e.g., 3 months).
        n_permutations (int): Number of permutations to perform.
        alternative (str): 'greater', 'less', or 'two-sided'.
        random_state (int or None): Random seed for reproducibility.
    
    Returns:
        p_value (float): P-value of the test.
        observed_stat (float): Observed mean difference.
        permuted_stats (np.array): Permuted statistics.
    """
    rng = np.random.default_rng(random_state)

    diff = np.asarray(diff)
    n = len(diff)
    n_blocks = n // block_size

    # truncate to full blocks
    diff = diff[:n_blocks * block_size]
    blocks = diff.reshape(n_blocks, block_size)

    observed_stat = np.mean(diff)
    permuted_stats = np.empty(n_permutations)

    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_blocks)
        flipped_blocks = blocks * signs[:, np.newaxis]
        permuted_stats[i] = np.mean(flipped_blocks.flatten())

    if alternative == 'greater':
        p_value = np.mean(permuted_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(permuted_stats <= observed_stat)
    elif alternative == 'two-sided':
        p_value = np.mean(np.abs(permuted_stats) >= abs(observed_stat))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    return p_value, observed_stat, permuted_stats


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
        figsize=(17, 12), 
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

        ax.coastlines(resolution="10m", linewidth=0.25)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.25)
        ax.add_feature(cfeature.LAND)
        ax.set_title(title)
        
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.25, linestyle="--", color="gray"
            )
        gl.xlocator = MultipleLocator(0.4)
        gl.ylocator = MultipleLocator(0.4)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.scatter(
            stations.lon, stations.lat, 
            c="limegreen", marker=7, s=7,
            linewidth=0.5, alpha=0.65,
            zorder=10
            )
        ax.scatter(
            22.08, 38.17,
            c="r", marker=7, s=7,
            linewidth=0.5, alpha=0.65,
            zorder=10
            )
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.04, 0.78])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("Temperature [°C]")
        
    #plt.tight_layout() 
    if save_figures == True:
        plt.savefig(f"outputs//multiplot{temporal_idx}-{target_resolution}deg.png", dpi=1500, bbox_inches="tight")
    plt.show() 


#%% per station stats extraction
dfLD = insitu.loc[insitu.index.isin(hd.valid_time.values), :]
dfHD = insitu.loc[insitu.index.isin(hd.valid_time.values), :]
dfSITE = insitu.loc[insitu.index.isin(hd.valid_time.values), :]
dfERROR_DIFF = insitu.loc[insitu.index.isin(hd.valid_time.values), :]
dfDOWN_MINUS_ERA = insitu.loc[insitu.index.isin(hd.valid_time.values), :]

'''
dfSLOPE = pd.DataFrame(
    columns=["slopeLD", "slope_stdLD", "slopeHD", "slope_stdHD"], 
    index=stations.index
    )
'''
'''
Pithanotata na yparxei kai grhgoroteros tropos
Alla autos douleuei sigoura kala
Prosoxh sta indexes, exei mismatch logw multiindex
sta dataarray derived products
me kamia lambda function mhpws??
'''

# RE MLK KANE TA ME ARRAYS TI GYFTIES EINAI AUTES
slopesLD = []
slopes_stdLD = []
interceptsLD = []
intercepts_stdLD = []
rvaluesLD = []

slopesHD = []
slopes_stdHD = []
interceptsHD = []
intercepts_stdHD = []
rvaluesHD = []

skewnesses = []
stats_wilcoxon = []
pvalues_wilcoxon = []

pvalues_permutation = []
pvalues_block_permutation = []


non_normal_diff_error_distributions = []
statistically_significant_diffs = []
significant_permutations = []

print("Performing IDW interpolation on LD & HD Data...")
for i in range(len(stations)):
    station = stations.iloc[i]
    
    station_code = station.iloc[1]
    station_name = station.iloc[2]
    
    station_location = ( station.iloc[3], station.iloc[4] )
    
    print(f"Station: {station_name} ({station_code})")
    
    seriesLD = idw_interpolation_across_time(
        target=station_location, dataarray=dataarrayLD
        ).to_series().round(decimals=2)
    
    seriesHD = idw_interpolation_across_time(
        target=station_location, dataarray=dataarrayHD
        ).to_series().round(decimals=2)
    
    seriesSITE = insitu.loc[
        insitu.index.isin(hd.valid_time.values), str(station_code)
        ]
    
    
    if visualize == True:
        plt.plot(hd.valid_time.values, seriesLD, linestyle="--")
        plt.plot(hd.valid_time.values, seriesSITE, c="r", alpha=0.75)
        plt.plot(hd.valid_time.values, seriesHD, c="k")
        plt.title(f'2m Temperature - {station_name} ({station_code}) - {target_resolution}°')
        
        plt.legend(
            ["ERA5-Land", "In-situ", "Downscaled"], prop={'size': 7}, 
            framealpha=0.3
            )
        
        plt.xticks(rotation=30)
        plt.ylabel("Temperature  [°C]")
        plt.grid()
        if save_figures == True:
            plt.savefig(f'outputs//timeseries-{station_code}-{station_name}-{target_resolution}deg.png', dpi=300, bbox_inches="tight")
        plt.show()
    
    
    if seriesLD.isna().all():
        print("Station outside ERA5-Land grid.")
        print("\n")
        slopeLD = np.nan
        slope_stdLD = np.nan
        interceptLD = np.nan
        intercept_stdLD = np.nan
        rvalueLD = np.nan
        
        slopeHD = np.nan
        slope_stdHD = np.nan
        interceptHD = np.nan
        intercept_stdHD = np.nan
        rvalueHD = np.nan
        
        skewness = np.nan
        stat_wilcoxon = np.nan
        pvalue_wilcoxon = np.nan
        
        pvalue_permutation = np.nan
        
        errorLD = np.nan
        errorHD = np.nan
        diff = np.nan
        non_normal_diff_error_distribution = "NaN"
        statistically_significant_diff = "NaN"
        significant_permutation = "NaN"
        
        down_minus_era = np.nan
        
    else:
        print("Station inside ERA5-Land grid.")
        slopeLD, interceptLD, rvalueLD, pvalueLD, slope_stdLD = linregress(
            seriesSITE, seriesLD
            )
        intercept_stdLD = linregress(
            seriesSITE, seriesLD
            ).intercept_stderr
        print("for ERA5-Land:")
        print(linregress(seriesSITE, seriesLD))
        slopeHD, interceptHD, rvalueHD, pvalueHD, slope_stdHD = linregress(
            seriesSITE, seriesHD
            )
        intercept_stdHD = linregress(
            seriesSITE, seriesHD
            ).intercept_stderr
        print(f"for Downscaled to {target_resolution}deg:")
        print(linregress(seriesSITE, seriesHD))
        print("\n")
        
        
        # wilcoxon signed rank test
        # absolute values to check precision/performance
        # non-absolute to check for biases
        errorLD = np.abs(seriesLD.values - seriesSITE.values)
        errorHD = np.abs(seriesHD.values - seriesSITE.values)
        
        # check the difference between downscaled and ERA5-Land
        # this is the stat we are gonna test
        # test normality, symmetry, autocorrelation
        down_minus_era = seriesHD.values - seriesLD.values
        # qq-plot for normality
        probplot(down_minus_era, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot - {station_name} ({station_code})")
        plt.show()
        
        #normality=0
        #assymetry=0
        #autocorrelation=0
        
        if shapiro(down_minus_era).pvalue<0.05:
            print("Difference between Timeseries NOT-NORMAL...")
        else:
            print("Data likely Normal...")
        
        if np.abs(yule_kendall_skewness(down_minus_era))>0.25:
            print(f"Yule-Kendall Assymetry NON-Negligible...")
        else:
            print("Non-Significant Yule-Kendall Assymetry...")
            
        if np.abs(skew(down_minus_era))>0.5:
            print(f"Skewness Assymetry NON-Negligible...")
            if np.abs(skew(down_minus_era))>1:
                print("STRONG Skewness Assymetry detected...")
        else:
            print("Non-Significant Skewness Assymetry...")
            
        if acf(down_minus_era)[1]>0.3: # or centered_confint[1]
            print(f"STRONG Lag-1 Autocorrelation detected...")
        
        
        print(f"Shapiro-Wilk Test for errors - {station_name} - {station_code}")
        shapiro_statLD, shapiro_pvalueLD = shapiro(errorLD)
        shapiro_statHD, shapiro_pvalueHD = shapiro(errorHD)
        if (shapiro_pvalueLD>0.05) & (shapiro_pvalueHD>0.05):
            print("Both LD and HD are likely normally distributed...")
            print("Paired t-test appropirate?")
        else:
            print("One or Both of LD and HD errors likely non-normal...")
            print("Paired t-test not approrirate?")
        print("\n")
        
        plot_acf(errorLD)
        plt.title(f"Autocorrelation LD: {station_name} - {station_code}")
        plt.show()
        plot_acf(errorHD)
        plt.title(f"Autocorrelation HD: {station_name} - {station_code}")
        plt.show()
        
        

        
        # LD-HD to check if downscaled is an improvement
        diff = np.round(errorLD - errorHD, 2)  # to avoid roundoff error
        
        # check skewness of diff
        skewness = skew(diff)
        
        # test normality of d
        stat_shapiro, pvalue_shapiro = shapiro(diff)
        if pvalue_shapiro < 0.05:
            print("difference of errors distribution not normal")
            non_normal_diff_error_distribution = "non_normal"
        else:
            non_normal_diff_error_distribution = "normal"
        
        
        if np.median(diff) > 0:  #case of statistically significant improvement
            print(f"median error diff in favor of downscaled for {station_name} ({station_code})...")
            print("checking for statistically significant improvement...")
            # doing a wilcoxon and a permutation test in each case...
            stat_wilcoxon, pvalue_wilcoxon = wilcoxon(diff, alternative="greater")
            if pvalue_wilcoxon <0.05:
                print("IMPROVEMENT detected by Wilcoxon...")
                statistically_significant_diff = "better"
            else:
                print("NO statisrical significance by Wilcoxon...")
                statistically_significant_diff = "none"
            
            pvalue_permutation = permutation_test(diff, alternative="greater")
            if pvalue_permutation <0.05:
                print("IMPROVEMENT detected by Permutation...")
                significant_permutation = "better"
            else:
                print("NO statisrical significance by Permutation...")
                significant_permutation = "none"
        
        if np.median(diff) < 0:  #case of statistically significant deterioration
            print(f"median error diff against downscaled for {station_name} ({station_code})...")
            print("checking for statistically significant deterioration...")
            stat_wilcoxon, pvalue_wilcoxon = wilcoxon(diff, alternative="less")
            if pvalue_wilcoxon <0.05:
                print("DETERIORATION detected by Wilcoxon...")
                statistically_significant_diff = "worse"
            else:
                print("NO statisrical significance by Wilcoxon...")
                statistically_significant_diff = "none"
            
            pvalue_permutation = permutation_test(diff, alternative="less")
            if pvalue_permutation <0.05:
                print("DETERIORATION detected by Permutation...")
                significant_permutation = "worse"
            else:
                print("NO statisrical significance by Permutation...")
                significant_permutation = "none"
        
        if np.median(diff) == 0:  #any practical significance?
            print(f"median error diff ZERO for {station_name} ({station_code})...")
            print("CAREFULLY INTERPRET TEST RESULTS FOR THIS STATION...")
            print("checking for statistically significant differences...")
            stat_wilcoxon, pvalue_wilcoxon = wilcoxon(diff, alternative="two-sided")
            if pvalue_wilcoxon <0.05:
                print("DIFFERENCES detected by Wilcoxon...")
                statistically_significant_diff = "CARE"
            else:
                print("NO statisrical significance by Wilcoxon...")
                statistically_significant_diff = "none"
            
            pvalue_permutation = permutation_test(diff, alternative="two-sided")
            if pvalue_permutation <0.05:
                print("DIFFERENCES detected by Permutation...")
                significant_permutation = "CARE"
            else:
                print("NO statisrical significance by Permutation...")
                significant_permutation = "none"
                
        
        # warn if skewness in diff's distribution is detected
        print(f"median: {np.median(diff):.3f}")
        print(f"mean: {np.mean(diff):.3f}")
        print(f"skewness: {skewness:.3f}")
        if np.abs(skewness) > 0.5:
            print("moderate skewness detected, assees wilcoxon's results carefully")
        print("\n")
        # most error diffs are not normal...
        #stat_ttest, pvalue_ttest = ttest_rel(errorHD, errorLD)
        
        #print(acf(diff,nlags=len(diff))[11])
        plot_acf(diff)
        plt.title(f"Diff Autocorrelation: {station_name} - {station_code}")
        plt.show()
        #print("\n")
        
        
        # scatter plot
        if visualize == True:
            # scatter plot per station
            xmin = np.min([
                seriesLD.min(), seriesHD.min(), seriesSITE.min()
                ])
            xmax = np.max([
                seriesLD.max(), seriesHD.max(), seriesSITE.max()
                ])
            x= np.linspace(np.floor(xmin), np.ceil(xmax), 100)
            yLD = slopeLD*x+interceptLD
            yHD = slopeHD*x+interceptHD
            
                
            plt.grid(linestyle="--", alpha=0.65)
            plt.scatter(seriesSITE, seriesLD, s=5, alpha=0.95, marker="D")
            plt.plot(x, yLD, c="blue", linestyle="--", alpha=0.75, linewidth=1.5)
            plt.scatter(seriesSITE, seriesHD, s=5, alpha=0.95, marker="D")
            plt.plot(x, yHD, c="brown", linestyle="--", alpha=0.95, linewidth=1.5)
            plt.ylabel("Modeled Temperature  [°C]")
            plt.xlabel("Insitu Temperature  [°C]")
            '''
            if interceptLD<=0:
                arithmetic_stringLD=""
            else:
                arithmetic_stringLD="+"
            if interceptHD<=0:
                arithmetic_stringHD=""
            else:
                arithmetic_stringHD="+"
            #"±"
            '''
            labelLD = make_legend_label(
                slopeLD, slope_stdLD, interceptLD, intercept_stdLD
                )
            labelHD = make_legend_label(
                slopeHD, slope_stdHD, interceptHD, intercept_stdHD
                )
            plt.legend(
                ["ERA5-Land", 
                 labelLD,
                 "Downscaled", 
                 labelHD],
                loc='upper left',
                fontsize=7
                )
            plt.plot(x, x, c="k", linestyle="--", alpha=0.75, linewidth=1.5)
            plt.title(f"Scatter Plot - {station_name} ({station_code}) - {target_resolution}°")
            plt.axis("square")
            if save_figures == True:
                plt.savefig(f"outputs//scatter-plot-monthly-{station_code}-{station_name}-{target_resolution}deg.png", dpi=500, bbox_inches="tight")
            plt.show()
            
            # ecdf plot per station
            sorted_era5land = ecdf(seriesLD)[0]
            sorted_downscaled = ecdf(seriesHD)[0]
            sorted_insitu = ecdf(seriesSITE)[0]

            cdf_era5land = ecdf(seriesLD)[1]
            cdf_downscaled = ecdf(seriesHD)[1]
            cdf_insitu = ecdf(seriesSITE)[1]
            
            plt.plot(sorted_era5land, cdf_era5land, linestyle="--")
            plt.plot(sorted_insitu, cdf_downscaled, c="r")
            plt.plot(sorted_downscaled, cdf_insitu, c="k")
            plt.legend(
                ["ERA5-Land", "In-situ", "Downscaled"], 
                framealpha=0.3
                )
            plt.grid()
            plt.title(f"Empirical CDF - {station_name} ({station_code}) - {target_resolution}°")
            plt.xlabel("Temperature  [°C]")
            plt.ylabel("Probability")
            if save_figures == True:
                plt.savefig(f"outputs//ecdf-temperatures-{station_code}-{station_name}-{target_resolution}deg.png", dpi=500)
            plt.show()
            
            
        print(f"Moving Block Bootstrapping Test - {station_name} - {station_code}")
        print("1:")
        #print(compare_autocorrelated_series(seriesHD, seriesLD, 12))
        print("2:")
        #print(compare_autocorrelated_series2(seriesHD, seriesLD, 12))
    
    
    
    plt.plot(diff)
    plt.title(f"Diff: {station_name} - {station_code}")
    plt.show()
    
    
    dfERROR_DIFF.loc[:,str(station_code)] = diff
    
    dfDOWN_MINUS_ERA.loc[:,str(station_code)] = down_minus_era

    slopesLD.append(slopeLD)
    slopes_stdLD.append(slope_stdLD)
    interceptsLD.append(interceptLD)
    intercepts_stdLD.append(intercept_stdLD)
    rvaluesLD.append(rvalueLD)

    slopesHD.append(slopeHD)
    slopes_stdHD.append(slope_stdHD)
    interceptsHD.append(interceptHD)
    intercepts_stdHD.append(intercept_stdHD)
    rvaluesHD.append(rvalueHD)
    
    skewnesses.append(skewness)
    stats_wilcoxon.append(stat_wilcoxon)
    pvalues_wilcoxon.append(pvalue_wilcoxon)
    
    non_normal_diff_error_distributions.append(non_normal_diff_error_distribution)
    statistically_significant_diffs.append(statistically_significant_diff)
    significant_permutations.append(significant_permutation)
    
    
    dfLD.loc[:,str(station_code)] = seriesLD.values
    dfHD.loc[:,str(station_code)] = seriesHD.values


stations["pvalue_wilcoxon"] = np.round(pvalues_wilcoxon, 5)
stations["statistical_significance_wilcoxon"] = statistically_significant_diffs
stations["statistical_significance_permutation"] = significant_permutations
stations["skewness"] = np.round(skewnesses, 3)
stations["error_diff_mean"] =  np.round(dfERROR_DIFF.mean(axis=0).values, 3)
stations["error_diff_median"] =  np.median(dfERROR_DIFF, axis=0)

stations["slopesLD"] = slopesLD
stations["slopesHD"] = slopesHD
stations["slopesLD"] = slopesLD
stations["slopes_stdLD"] = slopes_stdLD
stations["slopes_stdHD"] = slopes_stdHD
stations["interceptsLD"] = interceptsLD
stations["interceptsHD"] = interceptsHD
stations["intercepts_stdLD"] = intercepts_stdLD
stations["intercepts_stdHD"] = intercepts_stdHD
stations["interceptsHD"] = interceptsHD
stations["rvaluesLD"] = rvaluesLD
stations["rvaluesLD"] = rvaluesLD
stations["rvaluesHD"] = rvaluesHD


print("Wilcoxon statisical significance results:")
print(stations.statistical_significance_wilcoxon.value_counts())
print(f"Out of: {stations.statistical_significance_wilcoxon.shape[0]} stations in total.")
print("\n")

# save these to apply statistical tests later
dfERROR_DIFF.to_csv(f"diff_absolute_errors{target_resolution}.csv")
dfDOWN_MINUS_ERA.to_csv(f"diff_downscaled_to_era5land{target_resolution}.csv")
dfLD.to_csv(f"era5land_at_stations{target_resolution}.csv")
dfHD.to_csv(f"downscaled_at_stations{target_resolution}.csv")
dfSITE.to_csv(f"stations.csv")


#%% Linear Fitting and Distribution Visualization - axtarmas
# These below get stacked "horizontally" (per station first)
# if future_stack=True specified, dont specify dropna=False
dfLD_stacked = dfLD.stack(future_stack=True)
dfHD_stacked = dfHD.stack(future_stack=True)
dfSITE_stacked = dfSITE.stack(future_stack=True)


df_stacked = pd.concat([dfLD_stacked, dfHD_stacked, dfSITE_stacked], axis=1)
df_stacked.columns = ["idw", "downscaled", "insitu"]
df_stacked['month'] = df_stacked.index.get_level_values(0).month
df_stacked['station_code'] = df_stacked.index.get_level_values(1).astype(int)

mask = ~np.isnan(dfSITE_stacked) & ~np.isnan(dfLD_stacked)
df_stacked_mask = df_stacked.loc[mask,:]


slopeLD, interceptLD, r_valueLD, p_valueLD, slope_stdLD = linregress(
    df_stacked_mask.insitu, df_stacked_mask.idw
    )
intercept_stdLD = linregress(
    df_stacked_mask.insitu, df_stacked_mask.idw
    ).intercept_stderr

print("All data metrics:")
print(linregress(df_stacked_mask.insitu, df_stacked_mask.idw))
slopeHD, interceptHD, r_valueHD, p_valueHD, slope_stdHD = linregress(
    df_stacked_mask.insitu, df_stacked_mask.downscaled
    )
intercept_stdHD = linregress(
    df_stacked_mask.insitu, df_stacked_mask.downscaled
    ).intercept_stderr
print(linregress(df_stacked_mask.insitu, df_stacked_mask.downscaled))


# Doing it through statsmodels - same results as scipy.linregress
YLD = df_stacked_mask.idw
YHD = df_stacked_mask.downscaled
X = df_stacked_mask.insitu
X = sm.add_constant(X)

modelLD = sm.OLS(YLD, X)
modelHD = sm.OLS(YHD, X)

resultsLD = modelLD.fit()
resultsHD = modelHD.fit()

print("\nLinear regression parameters:")
print("For ERA5-Land:")
print(resultsLD.params)
print("For Downscaled:")
print(resultsHD.params)


if visualize == True:
    
    # Histogram for Temperature Distribution
    sns.histplot(df_stacked_mask.idw, color='blue', kde=True, label='ERA5-Land', bins=10, stat='density')
    sns.histplot(df_stacked_mask.downscaled, color='green', kde=True, label='Downscaled', bins=10, stat='density')
    sns.histplot(df_stacked_mask.insitu, color='brown', kde=True, label='In-situ', bins=10, stat='density')
    plt.title('Histogram of Temperatures')
    plt.xlabel('Temperature  [°C]')
    plt.ylabel('Density')
    plt.grid()
    plt.legend()
    if save_figures == True:
        plt.savefig(f"outputs//histogram-{target_resolution}deg.png", dpi=1000)
    plt.show()
    
    
    # Linear fitting between insitu and modeled temperatures
    xmin = np.min([
        df_stacked.idw.min(),
        df_stacked.downscaled.min(),
        df_stacked.insitu.min()
        ])
    xmax = np.max([
        df_stacked.idw.max(),
        df_stacked.downscaled.max(),
        df_stacked.insitu.max()
        ])
    x= np.linspace(np.floor(xmin), np.ceil(xmax), 100)
    yLD = slopeLD*x+interceptLD
    yHD = slopeHD*x+interceptHD
        
    plt.grid(linestyle="--", alpha=0.65)
    plt.scatter(df_stacked_mask.insitu, df_stacked_mask.idw, s=0.35, alpha=0.95)
    plt.plot(x, yLD, c="blue", linestyle="--", alpha=0.75, linewidth=1)
    plt.scatter(df_stacked_mask.insitu, df_stacked_mask.downscaled, s=0.25, alpha=0.75)
    plt.plot(x, yHD, c="brown", linestyle="--", alpha=0.95, linewidth=1)
    plt.ylabel("Modeled Temperature  [°C]")
    plt.xlabel("Insitu Temperature  [°C]")
    '''
    if interceptLD<=0:
        arithmetic_stringLD=""
    else:
        arithmetic_stringLD="+"
    if interceptHD<=0:
        arithmetic_stringHD=""
    else:
        arithmetic_stringHD="+"
        '''
    labelLD = make_legend_label(
        slopeLD, slope_stdLD, interceptLD, intercept_stdLD
        )
    labelHD = make_legend_label(
        slopeHD, slope_stdHD, interceptHD, intercept_stdHD
        )
    plt.legend(
        ["ERA5-Land", 
         labelLD,
         "Downscaled", 
         labelHD],
        fontsize=7
        )
    plt.plot(x, x, c="k", linestyle="--", alpha=0.75, linewidth=1)
    plt.title(f"Scatter Plot - All Stations - {target_resolution}°")
    plt.axis("square")
    if save_figures == True:
        plt.savefig(f"outputs//scatter-plot-monthly-{target_resolution}deg.png", dpi=500, bbox_inches="tight")
    plt.show()


# ECDFs
sorted_era5land = ecdf(df_stacked_mask.idw)[0]
sorted_downscaled = ecdf(df_stacked_mask.downscaled)[0]
sorted_insitu = ecdf(df_stacked_mask.insitu)[0]

cdf_era5land = ecdf(df_stacked_mask.idw)[1]
cdf_downscaled = ecdf(df_stacked_mask.downscaled)[1]
cdf_insitu = ecdf(df_stacked_mask.insitu)[1]

if visualize == True:
    plt.plot(sorted_era5land, cdf_era5land, linestyle="--")
    plt.plot(sorted_insitu, cdf_downscaled, c="r")
    plt.plot(sorted_downscaled, cdf_insitu, c="k")
    plt.legend(
        ["ERA5-Land", "In-situ", "Downscaled"], 
        framealpha=0.3
        )
    plt.grid()
    plt.title(f"Empirical CDF - All Stations - {target_resolution}°")
    plt.xlabel("Temperature  [°C]")
    plt.ylabel("Probability")
    if save_figures == True:
        plt.savefig(f"outputs//ecdf-temperatures-{target_resolution}deg.png", dpi=500)
    plt.show()
    
    
    # QQ plots
    plt.scatter(sorted_insitu, sorted_era5land, s=1)
    plt.scatter(sorted_insitu, sorted_insitu, s=1, c="k")
    plt.scatter(sorted_insitu, sorted_downscaled, s=1, alpha=0.75)
    plt.grid()
    plt.axis("square")
    plt.title(f'Q-Q Plot - All Stations - {target_resolution}°')
    plt.ylabel("Modeled Temperature  [°C]")
    plt.xlabel("Insitu Temperature  [°C]")
    plt.legend(
        ["ERA5-Land", "In-situ", "Downscaled"],
        fontsize=8.5,
        framealpha=0.3
        )
    if save_figures == True:
        plt.savefig(f"outputs//qq-plot-{target_resolution}deg.png", dpi=1000)
    plt.show()


#%% Performance Metrics
rmseLD = root_mean_squared_error(df_stacked_mask.insitu, df_stacked_mask.idw)
rmseHD = root_mean_squared_error(df_stacked_mask.insitu, df_stacked_mask.downscaled)
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


temporal_group_key = "month"
temporal_metrics = df_stacked[mask].groupby([temporal_group_key])[
    ["idw","downscaled","insitu"]
    ].apply(
    lambda group: compute_metrics(group, mask)
    )
temporal_metrics["idw_variance"] = df_stacked[mask].groupby(
    temporal_group_key)[["idw"]].var()
temporal_metrics["downscaled_variance"] = df_stacked[mask].groupby(
    temporal_group_key)[["downscaled"]].var()
temporal_metrics["insitu_variance"] = df_stacked[mask].groupby(
    temporal_group_key)[["insitu"]].var()

temporal_metrics = temporal_metrics.round(decimals=3)
temporal_metrics.to_excel(f"outputs//temporal_metrics-{target_resolution}deg.xlsx")


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

station_metrics = station_metrics.round(decimals=3)
station_metrics.to_excel(f"outputs//station_metrics-{target_resolution}deg.xlsx")


#%% Metrics Visualization
if visualize == True:
    for metric in ["rmse", "mae", "mbe", "ubrmse"]:
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
        
        if metric == "rmse":
            ylabel = "Error [°C²]"
        else:
            ylabel = "Error [°C]"
        plt.ylabel(ylabel)
        
        if metric == "mbe":
            plt.ylim(top=0)
        
        if metric == "ubrmse":
            plt.ylim(bottom=0)
        
        plt.show()
        
        
        plt.plot(range(len(temporal_metrics)), temporal_metrics[metric+"LD"])
        plt.plot(range(len(temporal_metrics)), temporal_metrics[metric+"HD"])
        plt.xticks(
            range(len(temporal_metrics)), temporal_metrics.index,
            rotation=35
            )
        plt.grid()
        plt.title(f"{metric.upper()} - Per Month")
        plt.legend(["ERA5-Land", "Downscaled"])
        plt.xlabel("Month")
        
        if metric == "rmse":
            ylabel = "Error [°C²]"
        else:
            ylabel = "Error [°C]"
        plt.ylabel(ylabel)
        
        if metric == "mbe":
            plt.ylim(top=0)
        
        if metric == "ubrmse":
            plt.ylim(bottom=0)
        
        plt.show()


#%% merge
stations_merged = station_metrics.merge(stations, right_on=stations.columns[1], left_index=True, how="right")

stations_merged.index = stations_merged.WMO_code
stations_merged.to_excel(f"statistical_significance_plus_metrics{target_resolution}deg.xlsx")

regression_columns = ["slopesLD", "slopesHD", 
                      "slopes_stdLD", "slopes_stdHD",
                      "interceptsLD", "interceptsHD",
                      "intercepts_stdLD", "intercepts_stdHD",
                      "rvaluesLD", "rvaluesHD"]
stations_merged[regression_columns].to_csv(f"regression_statistics{target_resolution}.csv")


#sys.stdout = sys.__stdout__
#log_file.close()


#%% Statistical tests and other metrics
'''
More tests to consider:
    Goodness-of-Fit to Insitu:
        Kolmogorov-Smirnov Test (done)
        Anderson-Darling Test
        Cramér-von Mises Test
        
    Comparing Means and Variances:
        T-Test (for means)
        F-Test or Levene’s Test (for variance)
        
    Closeness of Distributions:
        Kullback-Leibler Divergence
        Earth Mover’s Distance (done)
        
    Comparison between ERA5-Land and Downscaled:
        Paired T-Test or Wilcoxon Signed-Rank Test
        Bootstrap Methods
'''
'''
# 2 sided Kolmogorov-Smirnov Test
# Measures maximum difference between CDFs?
# Checking if two distributions are significantly different?
# if p-value<0.05 the modeled distributions are statistically different 
# from insitu, still though, smaller distances mean better match, which can
# have practical significance?
ks_era5land = ks_2samp(df_stacked_mask.insitu, df_stacked_mask.idw)
ks_downscaled = ks_2samp(df_stacked_mask.insitu, df_stacked_mask.downscaled)
ks_intercomparison = ks_2samp(df_stacked_mask.idw, df_stacked_mask.downscaled)

print("\n")
print("2-sided Kolmogorv-Smirnov Test:")
print("ERA5-Land vs Insitu:")
print(f"statistic: {ks_era5land.statistic:.4f}, p-value: {ks_era5land.pvalue:.4f}")
print("Downscaled vs Insitu:")
print(f"statistic: {ks_downscaled.statistic:.4f}, p-value: {ks_downscaled.pvalue:.4f}")
print("ERA5-Land vs Downscaled Intercomparison:")
print(f"statistic: {ks_intercomparison.statistic:.4f}, p-value: {ks_intercomparison.pvalue:.4f}")
'''

# Wasserstein Distance (or Earth Mover's Distance)
# Measures total work needed to move one distribution to another?
# Comparing overall shape & spread?
'''
Especially useful if both models give p-value<0.05?
Here we can define an improvement threshhold, like 20%
if wd_downscaled < 0.8*wd_era5land, then we did something...
'''
'''
wd_era5land = wasserstein_distance(
    df_stacked_mask.insitu, df_stacked_mask.idw
    )
wd_era5land_normalized = wd_era5land/df_stacked_mask.insitu.std()
wd_downscaled = wasserstein_distance(
    df_stacked_mask.insitu, df_stacked_mask.downscaled
    )
wd_downscaled_normalized = wd_downscaled/df_stacked_mask.insitu.std()
wd_intercomparison = wasserstein_distance(
    df_stacked_mask.idw, df_stacked_mask.downscaled
    )
print("\n")
print("Wasserstein Distances:")
print(f'ERA5-Land vs Insitu: {wd_era5land:.3f} °C '
      f'({wd_era5land_normalized:.3f} normalized)'
      )
print(f'Downscaled vs Insitu: {wd_downscaled:.3f} °C '
      f'({wd_downscaled_normalized:.3f} normalized)'
      )
print(f'ERA5-land vs Downscaled Intercomparison: {wd_intercomparison:.3f} °C')


# Kullback-Leibler Divergence
if visualize == True:
    # Measures information loss when approximating one distribution with another?
    # Checking how well a model represents the true distribution?
    # Use KL divergence if you want to measure how much information is lost 
    # when approximating Insitu with a model?
    plt.plot(range(len(df_stacked_mask)), 
             kl_div(df_stacked_mask.idw, df_stacked_mask.insitu),
             linewidth=1
             )
    plt.plot(range(len(df_stacked_mask)), 
             kl_div(df_stacked_mask.downscaled, df_stacked_mask.insitu),
             linewidth=1
             )
    plt.legend(["ERA5-Land", "Downscaled"])
    plt.grid()
    plt.title("Kullback-Leibler Divergence")
    plt.show()
'''


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