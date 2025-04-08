# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:13:57 2025

@author: yiann

for cds requests, its N-W-S-E

right now this code is only for monthly requests
needs lots of adjustments for daily and hourly data...
"""

import os
import cdsapi

extent = [41.8, 19.6, 35.8, 28.3]  #N-W-S-E
#single_year = 2021
#multiple_requests = False
years = list(range(1992,2023))
timescale = ['hourly', 'daily', 'monthly']


extent_string =  f'N{extent[0]}-W{extent[1]}-S{extent[-2]}-E{extent[-1]}'
   
subfolder = (
    f'extent__{extent_string}'
    )
folder = os.path.join("outputs", subfolder)
os.makedirs(folder, exist_ok=True)

filename = (
    f't2m-era5-land-{timescale[2]}__'
    f'{extent_string}__'
    f'Period{years[0]}-{years[-1]}'
    f'.nc'
    )
file_path = os.path.join(folder, filename)


dataset = f"reanalysis-era5-land-{timescale[2]}-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["2m_temperature"],
    "year": [str(x) for x in years],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": extent
}

print('\n')
print(f'requesting: {dataset}')
print(f'for variable: "2m_temperature"')
print(f'for time period: {years[0]}-{years[-1]}')
print('\n')

client = cdsapi.Client()
client.retrieve(dataset, request, file_path)

print('\nDone. Stored in:')
print(f"{file_path}")
print('\n')




