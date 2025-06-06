# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 12:28:35 2025

@author: Giannis
"""
import math
import numpy as np
import pandas as pd

from scipy.stats import shapiro
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator


stations = pd.read_csv(
    "Valid_HNMS_Stations_Info__N41.8-W19.6-S35.8-E28.3__Period1992-2022.csv"
    )

insitu = pd.read_csv("insitu.csv", index_col=[0], parse_dates=True)
era5land = pd.read_csv("era5land_at_stations.csv", index_col=[0], parse_dates=True)
df001 = pd.read_csv("downscaled_at_stations0.01.csv", index_col=[0], parse_dates=True)
df002 = pd.read_csv("downscaled_at_stations0.02.csv", index_col=[0], parse_dates=True)
df005 = pd.read_csv("downscaled_at_stations0.05.csv", index_col=[0], parse_dates=True)

regression001 = pd.read_csv("regression_statistics0.01.csv", index_col=[0])
regression002 = pd.read_csv("regression_statistics0.02.csv", index_col=[0])
regression005 = pd.read_csv("regression_statistics0.05.csv", index_col=[0])


diff005 = pd.read_csv("diff_downscaled_to_era5land0.05.csv", index_col=[0], parse_dates=True)
diff002 = pd.read_csv("diff_downscaled_to_era5land0.02.csv", index_col=[0], parse_dates=True)
diff001 = pd.read_csv("diff_downscaled_to_era5land0.01.csv", index_col=[0], parse_dates=True)


save_figures = True

#%% functions
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

def ecdf(data):
    '''
    Returns an array of the sorted data
    And the ecdf value for each sorted datapoint
    '''
    return np.sort(data), np.arange(1, len(data) + 1) / len(data)

def manual_qqplot(data, label=None, color=None):
    data = np.sort(np.asarray(data))
    n = len(data)

    # Theoretical quantiles
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical_q = norm.ppf(probs)

    # Plot
    plt.plot(theoretical_q, data, 'o', label=label, color=color)
    

def plot_multiple_acfs(datasets, labels, colors, nlags=20, alpha=0.05):
    n = len(datasets[0])
    
    plt.figure(figsize=(10, 6))
    
    for data, label, color in zip(datasets, labels, colors):
        acf_values, confint = acf(data, nlags=nlags, fft=True, alpha=0.05)
        confint_centered_upper = np.abs(confint[:,1]-acf_values)
        confint_centered_lower = -np.abs(confint[:,1]-acf_values)
        
        lags = np.arange(len(acf_values))
        
        for lag, val in zip(lags, acf_values):
            plt.vlines(lag, 0, val, colors=color, linewidth=1.5, alpha=0.85)
            
        plt.scatter(lags, acf_values, label=label, color=color)
        
        plt.fill_between(
            lags[1:], 
            confint_centered_lower[1:], 
            confint_centered_upper[1:], 
            color=color, alpha=0.1)


#%% visualization
timeseries = df001.index  # all files have the same index...

for i in range(len(stations)):
    i_station = stations.iloc[i]
    i_station_code = i_station.iloc[1]
    i_station_code_string = str(i_station_code)
    i_station_name = i_station.iloc[2]
    print(f"Station: {i_station_name} ({i_station_code})")
    
    
    station_era5land = era5land[i_station_code_string]
    station005 = df005[i_station_code_string]
    station002 = df002[i_station_code_string]
    station001 = df001[i_station_code_string]
    station_insitu = insitu[i_station_code_string]
    
    
    # i could also do it with .loc[i_station_code, "parameter"]
    # for parameter in regression001.columns (or any other regression df)
    slope_era5land = regression001.iloc[i,0]
    slope005 = regression005.iloc[i,1]
    slope002 = regression002.iloc[i,1]
    slope001 = regression001.iloc[i,1]
    
    slope_std_era5land = regression001.iloc[i,2]
    slope_std005 = regression005.iloc[i,3]
    slope_std002 = regression002.iloc[i,3]
    slope_std001 = regression001.iloc[i,3]
    
    intercept_era5land = regression001.iloc[i,4]
    intercept005 = regression005.iloc[i,5]
    intercept002 = regression002.iloc[i,5]
    intercept001 = regression001.iloc[i,5]
    
    intercept_std_era5land = regression001.iloc[i,6]
    intercept_std005 = regression005.iloc[i,7]
    intercept_std002 = regression002.iloc[i,7]
    intercept_std001 = regression001.iloc[i,7]

    
    #%% timeseries
    print(f"Plotting Timeseries...")
    plt.plot(timeseries, station_era5land, linestyle="--", alpha=0.8, linewidth=1)
    plt.plot(timeseries, station_insitu, c="r", alpha=0.8, linewidth=1)
    plt.plot(timeseries, station005, c="silver", alpha=0.8, linewidth=1)
    plt.plot(timeseries, station002, c="dimgray", alpha=0.8, linewidth=1)
    plt.plot(timeseries, station001, c="k", alpha=0.8, linewidth=1)
    plt.title(f'2m Temperature - {i_station_name} ({i_station_code})')
    
    legend = plt.legend(
        ["ERA5-Land", "In-situ", "0.05°", "0.02°", "0.01°"], prop={'size': 7}, 
        frameon=True,
        fontsize=10,
        framealpha=0.2,
        loc="lower left"
        )
    for text in legend.get_texts():
        text.set_alpha(0.7)
        text.set_fontsize(5.5)
    
    plt.xticks(rotation=30)
    plt.ylabel("Temperature  [°C]")
    plt.grid()
    if save_figures == True:
        plt.savefig(f'images//timeseries-{i_station_code}-{i_station_name}.png', dpi=300, bbox_inches="tight")
    plt.show()
    
    
    if station_era5land.isna().all():
        continue
    
    #%% scatter plots
    xmin = np.min([
        station_era5land.min(), station005.min(), station002.min(),
        station001.min(), station_insitu.min()
        ])
    xmax = np.max([
        station_era5land.max(), station005.max(), station002.max(),
        station001.max(), station_insitu.max()
        ])
    x= np.linspace(np.floor(xmin), np.ceil(xmax), 100)
    y_era5land = slope_era5land*x+intercept_era5land
    y005 = slope005*x+intercept005
    y002 = slope002*x+intercept002
    y001 = slope001*x+intercept001
    
    tmin = np.min([
        y005.min(), y002.min(), y001.min(), y_era5land.min(),
        station_insitu.min(), station_era5land.min(), x.min()
        ])
    tmax = np.max([
        y005.max(), y002.max(), y001.max(), y_era5land.max(),
        station_insitu.max(), station_era5land.max(), x.max()
        ])
    
    tmin = math.floor(tmin/2)*2
    tmax = math.ceil(tmax/2)*2
    
    
    label_era5land = make_legend_label(
        slope_era5land, slope_std_era5land, 
        intercept_era5land, intercept_std_era5land
        )
    label005 = make_legend_label(
        slope005, slope_std005, intercept005, intercept_std005
        )
    label002 = make_legend_label(
        slope002, slope_std002, intercept002, intercept_std002
        )
    label001 = make_legend_label(
        slope001, slope_std001, intercept001, intercept_std001
        )
    
    print("Generating Scatter Plots...")
    # scatter plot per station - 0.05
    plt.grid(linestyle="--", alpha=0.65)
    plt.scatter(station_insitu, station_era5land, s=5, alpha=0.95, marker="D")
    plt.plot(x, y_era5land, c="blue", linestyle="--", alpha=0.75, linewidth=1.5)
    plt.scatter(station_insitu, station005, s=5, alpha=0.95, marker="D")
    plt.plot(x, y005, c="brown", linestyle="--", alpha=0.95, linewidth=1.5)
    plt.ylabel("Modeled Temperature  [°C]")
    plt.xlabel("Insitu Temperature  [°C]")
    legend = plt.legend(
        ["ERA5-Land - 0.1°", 
         label_era5land,
         "Downscaled - 0.05°", 
         label005],
        loc='upper left',
        frameon=True,
        framealpha=0.7,
        fontsize=7
        )
    '''
    for text in legend.get_texts():
        text.set_alpha(0.7)
        text.set_fontsize(5.5)
    '''
    plt.plot(x, x, c="k", linestyle="--", alpha=0.75, linewidth=1.5)
    plt.title(f"{i_station_name} ({i_station_code}) - 0.05°")
    plt.axis("square")
    plt.xlim([xmin-2, tmax+2])
    plt.ylim([tmin, tmax])
    if save_figures == True:
        plt.savefig(f"outputs//scatter-plot-monthly-{i_station_code}-{i_station_name}-0.05deg.png", dpi=500, bbox_inches="tight")
    plt.show()
    
    # scatter plot per station - 0.02
    plt.grid(linestyle="--", alpha=0.65)
    plt.scatter(station_insitu, station_era5land, s=5, alpha=0.95, marker="D")
    plt.plot(x, y_era5land, c="blue", linestyle="--", alpha=0.75, linewidth=1.5)
    plt.scatter(station_insitu, station002, s=5, alpha=0.95, marker="D")
    plt.plot(x, y002, c="brown", linestyle="--", alpha=0.95, linewidth=1.5)
    plt.ylabel("Modeled Temperature  [°C]")
    plt.xlabel("Insitu Temperature  [°C]")
    legend = plt.legend(
        ["ERA5-Land - 0.1°", 
         label_era5land,
         "Downscaled - 0.02°", 
         label002],
        loc='upper left',
        frameon=True,
        framealpha=0.7,
        fontsize=7
        )
    '''
    for text in legend.get_texts():
        text.set_alpha(0.7)
        text.set_fontsize(5.5)
    '''
    plt.plot(x, x, c="k", linestyle="--", alpha=0.75, linewidth=1.5)
    plt.title(f"{i_station_name} ({i_station_code}) - 0.02°")
    plt.axis("square")
    plt.xlim([xmin-2, tmax+2])
    plt.ylim([tmin, tmax])
    if save_figures == True:
        plt.savefig(f"outputs//scatter-plot-monthly-{i_station_code}-{i_station_name}-0.02deg.png", dpi=500, bbox_inches="tight")
    plt.show()
    
    # scatter plot per station - 0.01
    plt.grid(linestyle="--", alpha=0.65)
    plt.scatter(station_insitu, station_era5land, s=5, alpha=0.95, marker="D")
    plt.plot(x, y_era5land, c="blue", linestyle="--", alpha=0.75, linewidth=1.5)
    plt.scatter(station_insitu, station001, s=5, alpha=0.95, marker="D")
    plt.plot(x, y001, c="brown", linestyle="--", alpha=0.95, linewidth=1.5)
    plt.ylabel("Modeled Temperature  [°C]")
    plt.xlabel("Insitu Temperature  [°C]")
    legend = plt.legend(
        ["ERA5-Land - 0.1°", 
         label_era5land,
         "Downscaled - 0.01°", 
         label001],
        loc='upper left',
        frameon=True,
        framealpha=0.7,
        fontsize=7
        )
    '''
    for text in legend.get_texts():
        text.set_alpha(0.7)
        text.set_fontsize(5.5)
    '''
    plt.plot(x, x, c="k", linestyle="--", alpha=0.75, linewidth=1.5)
    plt.title(f"{i_station_name} ({i_station_code}) - 0.01°")
    plt.axis("square")
    plt.xlim([xmin-2, tmax+2])
    plt.ylim([tmin, tmax])
    if save_figures == True:
        plt.savefig(f"outputs//scatter-plot-monthly-{i_station_code}-{i_station_name}-0.01deg.png", dpi=500, bbox_inches="tight")
    plt.show()
    
    
    #%% ecdf plot
    print("Generating ECDF Plots...")
    sorted_era5land = ecdf(station_era5land)[0]
    sorted005 = ecdf(station005)[0]
    sorted002 = ecdf(station002)[0]
    sorted001 = ecdf(station001)[0]
    sorted_insitu = ecdf(station_insitu)[0]

    ecdf_era5land = ecdf(station_era5land)[1]
    ecdf005 = ecdf(station005)[1]
    ecdf002 = ecdf(station002)[1]
    ecdf001 = ecdf(station001)[1]
    ecdf_insitu = ecdf(station_insitu)[1]
    
    plt.plot(sorted_era5land, ecdf_era5land, linestyle="--", alpha=0.8, linewidth=1)
    plt.plot(sorted_insitu, ecdf_insitu, c="r", alpha=0.8, linewidth=1)
    plt.plot(sorted005, ecdf005, c="silver", alpha=0.8, linewidth=1)
    plt.plot(sorted002, ecdf002, c="dimgray", alpha=0.8, linewidth=1)
    plt.plot(sorted001, ecdf001, c="k", alpha=0.8, linewidth=1)
    legend = plt.legend(
        ["ERA5-Land", "In-situ", "0.05°", "0.02°", "0.01°"], 
        frameon=True,
        framealpha=0.3,
        loc="upper left"
        )
    for text in legend.get_texts():
        text.set_alpha(0.9)
        text.set_fontsize(9)
        
    plt.grid()
    plt.title(f"Empirical CDF - {i_station_name} ({i_station_code})")
    plt.xlabel("Temperature  [°C]")
    plt.ylabel("Probability")
    if save_figures == True:
        plt.savefig(f"images//ecdf-temperatures-{i_station_code}-{i_station_name}.png", dpi=500)
    plt.show()
    
    
    #%% Q-Q plots
    print("Generating Q-Q Plots...")
    datasets = [diff005.iloc[:,i], diff002.iloc[:,i], diff001.iloc[:,i]]
    labels = ['Diff 0.05°', 'Diff 0.02°', 'Diff 0.01°']
    colors = ['silver', 'dimgray', 'k']
    
    plt.figure(figsize=(8, 6))
    for data, label, color in zip(datasets, labels, colors):
        manual_qqplot(data, label=label, color=color)
    # Reference line (45-degree)
    plt.plot([-2.2, 2.2], [-2.2, 2.2], 'r--', label='1:1')
    
    plt.xlabel('Theoretical Quantiles (Standard Normal)')
    plt.ylabel('Sample Quantiles')
    plt.title(f'Q-Q Plot - {i_station_name} ({i_station_code})')
    plt.legend()
    plt.grid(True)
    plt.axis("square")
    plt.tight_layout()
    plt.savefig(f'images/qqplot-{i_station_code}-{i_station_name}.png', dpi=500)
    plt.show()
    
    
    #%% autocorelation plots
    print("Generating Autocorrelation Plots...")
    plot_multiple_acfs(
        datasets, labels, colors, nlags=20, alpha=0.05
        )
    line = plt.axhline(y=0.3, linestyle='--', color='red', linewidth=1.2)
    line.set_alpha(0.5)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'ACF Comparison Across Datasets - {i_station_name} ({i_station_code})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'images/acf-plot-{i_station_code}-{i_station_name}.png', dpi=500)
    plt.show()





#%% loments
'''
# this givesthe same result as insitu
insitu2 = pd.read_csv("TG-m__N41.8-W19.6-S35.8-E28.3__Period1992-2022.csv", 
                     index_col=0, parse_dates=True
                     ) 
#insitu = sum[insitu2.loc[insitu2.index.isin(ds001.index.values), :]
'''

