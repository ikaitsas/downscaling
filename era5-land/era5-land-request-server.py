# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:31:22 2025

@author: yiann
"""

import os
import cdsapi

start_year = 1960
number_of_years = 6

extent = [38.2, 22, 38.1, 22.1]  #N-W-S-E
station = "pyrgaki"
variables = ["2m_temperature", "total_precipitation"]
timescale = "hourly"


station_folder = f"{station}-{timescale}--"
variables_folder = f"{'-'.join(variables)}"
full_path = os.path.join(station_folder, variables_folder)
os.makedirs(full_path, exist_ok=True)

years = list(range(start_year, start_year+number_of_years))
month_strings = [
    ["01", "02", "03", "04"],
    ["05", "06", "07", "08"],
    ["09", "10", "11", "12"]
    ]
year_part = ["pt1", "pt2", "pt3"]


for single_year in years:
    for month_period,part in zip(month_strings,year_part):
        file_name = (
            f"{station}-{timescale}-{str(single_year)}-"
            f"{part}.nc"
            )
        file_path = os.path.join(full_path, file_name)
        

        dataset = "reanalysis-era5-land"
        request = {
            "variable": variables,
            "year": str(single_year),
            "month": month_period,
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                "00:00", "01:00", "02:00",
                "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00",
                "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00",
                "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00",
                "21:00", "22:00", "23:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": extent
        }
        
        
        print('\n')
        print(f'requesting from:\n{dataset}')
        print(f"saving in:\n{file_path}")
        print("\n")
        
        
        client = cdsapi.Client()
        client.retrieve(dataset, request, file_path)
        
        print('\nDone. Stored in:')
        print(f"{file_path}")
        print('\n')




