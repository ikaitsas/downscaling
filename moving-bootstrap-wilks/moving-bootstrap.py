# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple

from scipy.stats import skew
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from scipy.stats import probplot
from scipy.stats import ttest_rel

from scipy.optimize import fsolve

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf

from arch.bootstrap import optimal_block_length
from arch.bootstrap import MovingBlockBootstrap
from arch.bootstrap import StationaryBootstrap

import seaborn as sns
import matplotlib.pyplot as plt


target_resolution = 0.01
era5land_name = f"data/era5land_at_stations.csv"
downscaled_name = f"data/downscaled_at_stations{target_resolution}.csv"

era5land = pd.read_csv(era5land_name, index_col=0, parse_dates=True)
downscaled = pd.read_csv(downscaled_name, index_col=0, parse_dates=True)
insitu = pd.read_csv("data/insitu.csv", index_col=0, parse_dates=True)

station_info = pd.read_excel("HNMS_Stations_Info.xlsx")
columns = downscaled.columns.astype(np.int64)
station_info = station_info[station_info.WMO_code.isin(columns)].reset_index(drop=True)


#%% useless(?) functions
def yule_kendall_skewness(data):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)  # Median
    q3 = np.percentile(data, 75)

    if q3 - q1 == 0:
        return np.nan  # Avoid division by zero if data is constant
    return (q3 + q1 - 2 * q2) / (q3 - q1)

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

'''
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
'''
def round_sigfig(x):
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))))


#%% used functions
def AR(series, alpha=0.05):
    # determine type of autoregression process...
    if np.isnan(series).sum() >0:
        AR = -999
    else:
        acf_vals, confint = acf(series, nlags=2, fft=False, alpha=alpha)
        confint_centered = np.abs(confint[:,1]-acf_vals)
        rho1 = acf_vals[1]
        rho2 = acf_vals[2]
        
        # determine the type of AR process
        if rho1>confint_centered[1] and rho2>confint_centered[2]:
            AR = 2
        # not sure about this, apla to mantewuw...
        elif rho1<confint_centered[1] and rho2>confint_centered[2]:
            AR = 2
        elif rho1>confint_centered[1] and rho2<confint_centered[2]:
             AR = 1 
        elif rho1<confint_centered[1] and rho2<confint_centered[2]:
             AR = 0
        else:
            AR = 0
            
    return int(AR) if AR>-1 else np.nan

def vif_wilks_AR(series, AR=1, adjusted=True, alpha=0.05):
    # Uses AR1 or AR2 to model the timeseries rhos
    if AR == 0:
        #print("no need for vif...")
        vif=1
    
    n = len(series)
    acf_vals = acf(series, nlags=n-1, fft=False, alpha=None)
    rho1 = acf_vals[1]
    rho2 = acf_vals[2]
    
    if AR==1:
        # Compute weighted sum of autocorrelations
        k = np.arange(1, n)  # Lags from 1 to n-1
        weights = 1 - k / n  # Finite-sample weights (1 - k/n)
        rhos = rho1 ** k     # AR(1) autocorrelations (rho_k = rho1^k)
    
        # Wilks' exact VIF formula
        vif = 1 + 2 * np.sum(weights * rhos)
        
        if adjusted == True:
            vif = vif * np.exp(2*vif/n)
    
    if AR == 2:
        rhos = np.zeros(n-1)
        phi1 = rho1 * (1-rho2)/(1-rho1**2)
        phi2 =  (rho2-rho1**2)/(1-rho1**2)
        rhos[0], rhos[1] = rho1, rho2
        for k_rho in range(2,n-1):
            rhos[k_rho] = phi1*rhos[k_rho-1] + phi2*rhos[k_rho-2]
        
        k = np.arange(1, n)  # Lags from 1 to n-1
        weights = 1 - k / n  # Finite-sample weights (1 - k/n)
        vif = 1 + 2 * np.sum(weights * rhos)
    
    if AR>2:
        print("mh vazeis malakies inputs...")
        vif = np.nan
        
        if adjusted == True:
            vif = vif * np.exp(3*vif/n)
            
    return vif


#%% find optimal block size functions - per wilks
def equation_for_block_size_AR1_wilks(l, n, vif):
    exponent = (2/3) * (1 - 1/vif)
    return l - (n - l - 1)**exponent

def find_optimal_block_size_AR1_wilks(n, vif, l_guess=None):
    if l_guess == None:
        l_guess = np.sqrt(n)
    # Use fsolve to find root of the equation
    result, info, ier, msg = fsolve(equation_for_block_size_AR1_wilks, l_guess, args=(n, vif), full_output=True)
    if ier == 1:
        return result[0]
    else:
        raise ValueError(f"Root finding did not converge: {msg}")
        
# find optimal block size - per wilks
def equation_for_block_size_AR2_wilks(l, n, vif):
    exponent = (2/3) * (1 - 1/np.sqrt(4*vif))
    return l - (n - l - 1)**exponent

def find_optimal_block_size_AR2_wilks(n, vif, l_guess=None):
    if l_guess == None:
        l_guess = np.sqrt(n)
    # Use fsolve to find root of the equation
    result, info, ier, msg = fsolve(equation_for_block_size_AR2_wilks, l_guess, args=(n, vif), full_output=True)
    if ier == 1:
        return result[0]
    else:
        raise ValueError(f"Root finding did not converge: {msg}")

def optimal_block_size_wilks(series):
    n = len(series)
    AR_series = AR(series)
    #print(f"lenght: {n}")
    #print(f"AR: {AR_series}")
    
    if AR_series == 0:
        optimal_block_size = 1
    
    if AR_series == 1:
        vif_series = vif_wilks_AR(series, AR=AR_series)
        optimal_block_size = find_optimal_block_size_AR1_wilks(n, vif=vif_series)
    
    if AR_series == 2:
        vif_series = vif_wilks_AR(series, AR=AR_series)
        optimal_block_size = find_optimal_block_size_AR2_wilks(n, vif=vif_series)
    
    optimal_block_size = int( np.round(optimal_block_size) )
    #print(f"optimal block size: {optimal_block_size}")
        
    return optimal_block_size


#%% testing functions
def jackknife_after_bootstrap_variance(resampled, block_size):
    """
    Calculate Jackknife-After-Bootstrap (JAB) variance estimate.

    Parameters:
    - resampled: numpy array of resampled data
    - block_size: integer size of each block
    - n: original data size (before resampling)

    Returns:
    - JAB_var: variance estimate
    """
    n = len(resampled)
    remaining = len(resampled) % block_size
    num_full_blocks = len(resampled) // block_size

    if remaining != 0:
        resampled_jacknife = resampled[:-remaining]
    else:
        resampled_jacknife = resampled
    
    blocks = []
    for i in range(num_full_blocks):
        blocks.append(resampled_jacknife[i*block_size:(i+1)*block_size])
        
    jacknife_mean_values = []
    # Leave-one-out for full blocks only
    for i in range(num_full_blocks):
        # Combine all blocks except the i-th full block
        blocks_jacknife = blocks[:i] + blocks[i+1:]
        sum_per_jacknife_block = [np.sum(x) for x in blocks_jacknife]
        sum_of_jacknife_sums = np.sum(sum_per_jacknife_block)
        jacknife_mean_value = sum_of_jacknife_sums/(num_full_blocks*block_size - block_size)
        jacknife_mean_values.append(jacknife_mean_value)
    
    jacknife_mean_values = np.array(jacknife_mean_values)
    jacknife_mean_of_means = np.sum(jacknife_mean_values)/num_full_blocks
    
    # jacknife-after-bootstrap variance calculation
    jacknife_tetragwna = np.sum((jacknife_mean_values-jacknife_mean_of_means)**2)
    JAB_var = (num_full_blocks*block_size/n)*((num_full_blocks-1)/num_full_blocks)*jacknife_tetragwna
    
    return JAB_var

# to parakatw einai to idio alla trexei 50-100 fores grhgorotera
def jackknife_after_bootstrap_variance2(resampled, block_size):
    """
    Efficient Jackknife-After-Bootstrap (JAB) variance estimator.
    Approximately 50 - 100 time faster i think...
    """
    resampled = np.asarray(resampled)

    n = len(resampled)
    num_full_blocks = n // block_size
    trimmed_length = num_full_blocks * block_size
    
    if num_full_blocks < 2:
        # Not enough blocks to perform jackknife
        return 0.0

    # Trim to full blocks only
    resampled = resampled[:trimmed_length]

    # Reshape into (num_blocks, block_size)
    blocks = resampled.reshape(num_full_blocks, block_size)

    # Precompute block sums
    block_sums = blocks.sum(axis=1)

    # Total sum of all blocks
    total_sum = block_sums.sum()

    # Leave-one-block-out means
    loo_sums = total_sum - block_sums  # each LOO sum
    loo_counts = trimmed_length - block_size
    jackknife_means = loo_sums / loo_counts

    # Mean of jackknife means
    mean_jackknife = jackknife_means.mean()

    # Jackknife-after-bootstrap variance estimate
    jacknife_tetragwna = np.sum((jackknife_means-mean_jackknife)**2)
    JAB_var = (num_full_blocks*block_size/n)*((num_full_blocks-1)/num_full_blocks)*jacknife_tetragwna
    
    return JAB_var

def moving_block_bootstrap_wilks(
        x, y, n_bootstrap=10000, block_size=None,
        S0=0, alternative="two-sided"
        ):
    # moving block bootstrap for differnece of means
    # based on wilks 1997 paper:
        # Resampling Hypothesis Tests for Autocorrelated Fields
    # the AR process (1 or 2) is determined inside here
    # same as the optimal block length
    
    # NO DIRECT IMPLEMENTATION OF VIF=1
    # MUST BE TESTED IF IT WORKS FINE FOR VIF=1
    # i think it works... # later comment
    
    # its for the difference S = mean(x) - mean(y) == S0
    # S0=0 is the null hypothesis that the means are the same
    # classic alternatives, two-sided, greater, less included
    
    # remember that the resulting p-value is an estimate, and can
    # still be subject to randomness, running this bootstrap test
    # multiple times (10-100?) on the same data pair should be 
    # considered, or even using a stationary bootstrap implementation...
    
    vif_x = vif_wilks_AR(x,AR=AR(x))
    vif_y = vif_wilks_AR(y,AR=AR(e5))
    
    S = np.mean(x) - np.mean(y)
    S_variance = np.sqrt( vif_x*x.var()/len(x) + vif_y*y.var()/len(y) )
    statistic = (S-S0)/S_variance

    
    if block_size == None:
        block_size_x = optimal_block_size_wilks(x)
        block_size_y = optimal_block_size_wilks(y)
    else:
        block_size_x = block_size
        block_size_y = block_size
    
    #print(block_size_x, block_size_y)
        
    # bootsptrapping each sample separetely, as per wilks (1997)
    bs_x = MovingBlockBootstrap(block_size_x, x)#, seed=42)
    bs_y = MovingBlockBootstrap(block_size_y, y)#, seed=69)
    
    #jab_variances = []
    #Ss_bootstrap = []
    statistics_bootstrap = []
    for sample_x, sample_y in zip(bs_x.bootstrap(n_bootstrap), bs_y.bootstrap(n_bootstrap)):
        #print(bs_x.index)
        #print(",")
        #print(bs_y.index)
        #print("\n")
        x_resampled = sample_x[0][0]
        y_resampled = sample_y[0][0]
        
        # jacknife-after-bootstrap variance calculation for the 2 samples
        x_jab_variance = jackknife_after_bootstrap_variance2(x_resampled, block_size=block_size_x)
        y_jab_variance = jackknife_after_bootstrap_variance2(y_resampled, block_size=block_size_y)
        
        jab_variance = np.sqrt(x_jab_variance + y_jab_variance)
        #jab_variances.append(jab_variance)
        
        S_bootstrap = np.mean(x_resampled) - np.mean(y_resampled)
        #Ss_bootstrap.append(S_bootstrap)

        if jab_variance == 0:
            statistic_bootstrap = np.nan
        else:
            statistic_bootstrap = (S_bootstrap - S)/jab_variance
        
        
        statistics_bootstrap.append(statistic_bootstrap)

    statistics_bootstrap = np.array(statistics_bootstrap)
    
    # sloppy failsafe for if there is a nan value is the statistics
    if np.isnan(statistics_bootstrap).any() == True:
        statistics_bootstrap = [~np.isnan(statistics_bootstrap)]
        n_bootstrap = len(statistics_bootstrap)

    # adjust for two-sided, grater or less scenarios
    # might nedd a +1 on the numerator, to avoid pvalue=0 cases
    if alternative == "two-sided":
        pvalue = ( (np.abs(statistics_bootstrap)>np.abs(statistic)).sum()+1 ) / (n_bootstrap+1)
    elif alternative == "greater":
        pvalue = ( (statistics_bootstrap>statistic).sum()+1 ) / (n_bootstrap+1)
    elif alternative == "less":
        pvalue = ( (statistics_bootstrap<statistic).sum()+1 ) / (n_bootstrap+1)
    else:
        print("returned pvalue 1.0, insert correct alternative option")
        pvalue = 1.0

    #print(f"p-value: {pvalue}")
    
    return statistic, pvalue


#%% misc functions
def organize_testing(x, y):
    sign_of_difference_of_means = (
        x.mean()-y.mean()
        )/np.abs(
            x.mean()-y.mean()
            )
    
    # one-sided testing
    if np.isnan(x).sum()>0:
        os_statistic = np.nan
        os_pvalue = np.nan
    else:
        if sign_of_difference_of_means>0:
            os_statistic, os_pvalue = moving_block_bootstrap_wilks(
                x, y, alternative="greater"
                )
        if sign_of_difference_of_means<0:
            os_statistic, os_pvalue = moving_block_bootstrap_wilks(
                x, y, alternative="less"
                )
        if sign_of_difference_of_means==0:
            os_statistic, os_pvalue = 0, 1.0
    
    print(f"one-sided statistic: {os_statistic:.5f}, p-value: {os_pvalue:.5f}")
            
    # two-sided testing
    if np.isnan(x).sum()>0:
        ts_statistic = np.nan
        ts_pvalue = np.nan
    else:
        ts_statistic, ts_pvalue = moving_block_bootstrap_wilks(
            x, y, alternative="two-sided"
            )
    
    print(f"two-sided statistic: {ts_statistic:.5f}, p-value: {ts_pvalue:.5f}")
    
    
    return os_pvalue, ts_pvalue, sign_of_difference_of_means


#%% testing with bootstrap at all stations
os_pvalues = []
ts_pvalues = []
os_pvalues_error = []
ts_pvalues_error = []
signs = []
signs_error = []
for i in range(len(downscaled.columns)):
    station_code = int(downscaled.columns[i])
    station_name = station_info.iloc[i,1]
    print(f"station: {station_name} - {station_code}")
    
    print("Differences between downscaled and era5-land...")
    ds = downscaled.iloc[:,i]
    e5 = era5land.iloc[:,i]
    situ = insitu.iloc[:,i]
    
    os_pvalue, ts_pvalue, sign_of_difference_of_means = organize_testing(ds, e5)
    os_pvalues.append(os_pvalue)
    ts_pvalues.append(ts_pvalue)
    signs.append(sign_of_difference_of_means)
    
    print("Differences between absolute errors downscaled and era5-land...")
    diff_ds = np.abs(ds - situ)
    diff_e5 = np.abs(e5 - situ)
    
    os_pvalue_error, ts_pvalue_error, sign_of_difference_of_means_error = organize_testing(diff_ds, diff_e5)
    os_pvalues_error.append(os_pvalue_error)
    ts_pvalues_error.append(ts_pvalue_error)
    signs_error.append(sign_of_difference_of_means_error)
    
    print("\n")


station_info["ds-e5_one_sided"] = os_pvalues
station_info["ds-e5_sign"] = signs
station_info["ds-e5_two_sided"] = ts_pvalues

station_info["error_ds-error_e5_one_sided"] = os_pvalues_error
station_info["error_ds-error_e5_sign"] = signs_error
station_info["error_ds-error_e5_two_sided"] = ts_pvalues_error


station_info = station_info.drop(columns=["Comments"])
station_info.to_excel(f"statistical_significance_bootstrap{target_resolution}.xlsx")


#%% lets play
'''
i=8  # 8: Konitsa
station_code = int(downscaled.columns[i])
print(f"station: {station_info.iloc[i,1]} - {station_code}")

ds = downscaled.iloc[:,i]
e5 = era5land.iloc[:,i]
situ = insitu.iloc[:,i]

diff_ds = np.abs(ds - situ)
diff_e5 = np.abs(e5 - situ)

sing_of_difference_of_means = ds.mean()-e5.mean()

statistic, pvalue = moving_block_bootstrap_wilks(ds, e5, alternative="two-sided")

dstatistic, dpvalue = moving_block_bootstrap_wilks(diff_ds, diff_e5, alternative="two-sided")

print(f"differences - statistic: {statistic:.5f}, p-value: {pvalue:.5f}")
print(f"differences of absolute errors - statistic: {dstatistic:.5f}, p-value: {dpvalue:.5f}")
'''

'''
for i in range(len(downscaled.columns)):
    station_code = int(downscaled.columns[i])
    ds = downscaled.iloc[:,i]
    e5 = era5land.iloc[:,i]
    diff_ds = np.abs(ds - situ)
    diff_e5 = np.abs(e5 - situ)
    AR_ds = AR(diff_ds)
    AR_e5 = AR(diff_e5)
    
    print(station_code, AR_ds, AR_e5)
'''

'''
import timeit
orig_time = timeit.timeit(
    stmt="jackknife_after_bootstrap_variance2(diff_ds, block_size=7)",
    globals=globals(),
    number=100
)
print(f"Original average time (100 runs): {orig_time / 100:.9f} seconds")
'''

