# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:30:10 2025

@author: yiann

yparxoun 25 stathmoi apo autous sto nms-data sto euros 39-36N,21-24E
"""

import os
import numpy as np
import pandas as pd

time_scale = 0  #0:Monthly, 1:Daily
parameter = 1  #0:RR, 1:TG, 2:TN, 3:TX
station = 16606  #other options available


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
df = pd.read_csv(path, index_col=0)
