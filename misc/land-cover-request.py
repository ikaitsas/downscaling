# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:13:57 2025

@author: yiann

CDR and ICDR Sentinel-3 Land Cover classifications CDS request script

guides and specifications (v2.1 & v2.0 respectively) at:
https://dast.copernicus-climate.eu/documents/satellite-land-cover/WP2-FDDP-LC-2021-2022-SENTINEL3-300m-v2.1.1_PUGS_v1.1_final.pdf
https://dast.copernicus-climate.eu/documents/satellite-land-cover/D3.3.11-v1.0_PUGS_CDR_LC-CCI_v2.0.7cds_Products_v1.0.1_APPROVED_Ver1.pdf

for cds requests, extent is N-W-S-E
for global LC maps, set extent to [90, -180, -90, 180]

outputs are stored in the outputs folder, in the corresponding extent subfolder
"""

import os
import cdsapi

extent = [39, 21, 36, 24]
single_year = 2021
multiple_requests = False
years = list(range(1992,2023))


#%% first steps
def filter_and_check(lst, min_val, max_val):
    # Find elements that are out of range
    smaller = [x for x in lst if x < min_val]
    larger = [x for x in lst if x > max_val]

    # Filtered list within range
    filtered = [x for x in lst if min_val <= x <= max_val]

    # Print warnings if out-of-range elements exist
    if smaller:
        print('Dataset starts at 1992. Using 1992 as year...')
    if larger:
        print('Dataset ends at 2022. Using 2022 as last year...')

    return filtered

if multiple_requests == True:
    years = filter_and_check(years, 1992, 2022)
    print(f'Multiple requests, range: {years[0]}-{years[-1]}...')
else:
    years = [single_year]
    print(f'Single request, year: {years[0]}')


# store the requested datasets in a folder/subfolder structure
# an extent file is created, inside it the years are stored
if extent == [90, -180, -90, 180]:
    extent_string = "Global"
else:
   extent_string =  f'N{extent[0]}-W{extent[1]}-S{extent[-2]}-E{extent[-1]}'
   
subfolder = (
    f'extent__{extent_string}'
    )
folder = os.path.join("outputs", subfolder)
os.makedirs(folder, exist_ok=True)


#%% starting request loop
for year in years:
    print(f'Requesting data for: {year}...')
    
    # give the requested data filenames
    filename = (
        f'land-cover__'
        f'{extent_string}__'
        f'Period{year}'
        f'.nc'
        )
    file_path = os.path.join(folder, filename)
    
    
    # check if file already exists
    if os.path.exists(file_path):
        print(f"File already exists:\n{file_path}")
        print("Skipping download.")
        print('\n')
        
    # if nonexistant, proceed with the request 
    else:
        print("File not found. Proceeding with data request...")
        
        print(f"Year:  {year}")
        if year < 2016:
            print('Dataset version: v2_0_7cds')
            version = "v2_0_7cds"
            
        if year >= 2016:
            print('Dataset version: v2_1_1')
            version = "v2_1_1"
            
        dataset = "satellite-land-cover"
        request = {
            "variable": "all",
            "year": [f"{year}"],
            "version": [version],
            "area": extent
        }
        client = cdsapi.Client()
        client.retrieve(dataset, request, file_path)
        
        print('\nDone. Stored in:')
        print(f"{file_path}")
        print('\n')


print('Finally done.')




'''
try:
    print(year)
    if year < 1992:
        print('Dataset starts at 1992. Using 1992 as year...')
        year = 1992
    
    if year > 2022:
        print('Dataset ends at 1992. Using 2022 as year...')
        year = 2022
except: NameError
'''
