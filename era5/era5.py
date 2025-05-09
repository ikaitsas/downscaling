# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:12:31 2025

@author: Giannis
"""

import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker



era5_file = "t2m.nc"
dsLD = xr.open_dataset(era5_file)
 


