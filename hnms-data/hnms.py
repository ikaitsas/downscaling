# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:30:10 2025

@author: yiann

yparxoun 25 stathmoi apo autous sto nms-data sto euros 39-36N,21-24E
alla oi 20 peftoun mesa se non-NaN gridpoint tou ERA5-Land...
"""

import os
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

time_scale = 0  #0:Monthly, 1:Daily
parameter = 1 #0:RR, 1:TG, 2:TN, 3:TX
station = 16606  #other options available

extent = [39, 21, 36, 24]  #N-W-S-E
year_start = 1992
year_end = 2022


parameters = ["RR", "TG", "TN", "TX"]
temporal_resolution = {
    "Monthly":"m",
    "Daily":"d"
    }
temporal_resolution = tuple(temporal_resolution.items())


#%%find filename based on above query orders
if time_scale == 0:
    file = (
        f'{parameters[parameter]}-'
        f'{temporal_resolution[time_scale][1]}_'
        f'1960-2022_series.csv'
    )
    
    path = os.path.join(os.getcwd(), 
                        "Homogenized", 
                        str(temporal_resolution[time_scale][0]), 
                        str(file))

if time_scale == 1:
    file = (
        f'{station}_{parameters[parameter]}_'
        f'{temporal_resolution[time_scale][1]}_h.csv'
        )
    
    path = os.path.join(os.getcwd(), 
                        "Homogenized", 
                        str(temporal_resolution[time_scale][0]),
                        f"{parameters[parameter]}",
                        str(file))


#%% pandas
df = pd.read_csv(path, index_col=0, parse_dates=True)

hnms = pd.read_excel("HNMS_Stations_Info.xlsx")

hnms_extent = hnms[(hnms.iloc[:,3]>=extent[1]) & #W
                   (hnms.iloc[:,3]<extent[-1]) & #E
                   (hnms.iloc[:,2]>extent[-2]) & #S
                   (hnms.iloc[:,2]<=extent[0])   #N
                   ]


'''
the following are designed for the monthly homogenized values...
the daily homogenized and the daily raw most probably need
some other kind of handling, like concatenation on columns, from what
i can at least interpret from the .csv files
'''

if df.index.year.max()<year_end:
    print('\n')
    print(f'End year {year_end} too large. Using dataset end year...')
    year_end = df.index.year.max()
    print(f'End year now is: {year_end}')

if df.index.year.min()>year_start:
    print('\n')
    print(f'Start year {year_start} too small. Using dataset start year...')
    year_start = df.index.year.min()
    print(f'Start year now is: {year_start}')


# isolate the parameters and HNMS that are common in both dataframes
# based on the imported/given initial values
df_extent = df.loc[:, df.columns.isin(
    hnms_extent.WMO_code.astype(str).to_list() )
    ]
df_extent = df_extent[
    (df_extent.index.year >= year_start) & 
    (df_extent.index.year <= year_end) 
    ]

if df_extent.empty:
    hnms_extent = pd.DataFrame( columns=hnms_extent.columns )
    print('No matching columns were found...')
else:
    hnms_extent = hnms_extent[
        hnms_extent.WMO_code.astype(str).isin(df_extent.columns)
        ]


#%% store files in respective folders
extent_string = f"N{extent[0]}-W{extent[1]}-S{extent[-2]}-E{extent[-1]}"

subfolder = (
    f'extent__{extent_string}__'
    f'Period{year_start}-{year_end}'
    )
subfolder_Path = os.path.join("outputs-storage", subfolder)

os.makedirs(subfolder_Path, exist_ok=True)
# maybe should add invalid character handling in subfolder names


df_extent_File = (
    f'{parameters[parameter]}-{temporal_resolution[0][1]}__'
    f'{extent_string}__'
    f'Period{year_start}-{year_end}'
    f'.csv'
    )
df_extent_Path = os.path.join(subfolder_Path, df_extent_File)
df_extent.to_csv(df_extent_Path)

hnms_extent_File = (
    f'Valid_HNMS_Stations_Info__'
    f'{extent_string}__'
    f'Period{year_start}-{year_end}'
    f'.csv'
    )
hnms_extent_Path = os.path.join(subfolder_Path, hnms_extent_File)
hnms_extent.to_csv(hnms_extent_Path)


