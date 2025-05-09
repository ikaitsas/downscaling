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


station = 16606  #other options available
extent = [41.8, 19.6, 35.8, 28.3]  #N-W-S-E
coarse_resolution = 0.1  #degrees
year_start = 1992
year_end = 2022

file_path = "TG-m_1960-2022_series.csv"
station_info_path = "HNMS_Stations_Info.xlsx"


#%% pandas
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

hnms = pd.read_excel(station_info_path)

hnms_extent = hnms[(hnms.iloc[:,3]>=extent[1]-coarse_resolution/2) & #W
                   (hnms.iloc[:,3]<extent[-1]+coarse_resolution/2) & #E
                   (hnms.iloc[:,2]>extent[-2]-coarse_resolution/2) & #S
                   (hnms.iloc[:,2]<=extent[0]+coarse_resolution/2)   #N
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
    f'TG-m__'
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


