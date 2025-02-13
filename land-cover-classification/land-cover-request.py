# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:13:57 2025

@author: yiann

CDR and ICDR Sentinel-3 Land Cover classifications CDS request script

guides and specifications (v2.1 & v2.0 respectively) at:
https://dast.copernicus-climate.eu/documents/satellite-land-cover/WP2-FDDP-LC-2021-2022-SENTINEL3-300m-v2.1.1_PUGS_v1.1_final.pdf
https://dast.copernicus-climate.eu/documents/satellite-land-cover/D3.3.11-v1.0_PUGS_CDR_LC-CCI_v2.0.7cds_Products_v1.0.1_APPROVED_Ver1.pdf
"""

import cdsapi

'''
#for v2.1, years >=2016
dataset = "satellite-land-cover"
request = {
    "variable": "all",
    "year": ["2022"],
    "version": ["v2_1_1"],
    "area": [39, 21, 36, 24]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()


#for v2.0, years <=2015
dataset = "satellite-land-cover"
request = {
    "variable": "all",
    "year": ["2002"],
    "version": ["v2_0_7cds"],
    "area": [39, 21, 36, 24]
}

filename = "=whatever.nc"
client = cdsapi.Client()
client.retrieve(dataset, request, filename).download()


#if both versions present:
dataset = "satellite-land-cover"
request = {
    "variable": "all",
    "year": ["2015", "2016"],
    "version": [
        "v2_0_7cds",
        "v2_1_1"
    ],
    "area": [39, 21, 36, 24]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
'''



