# -*- coding: utf-8 -*-
"""
Created on Fri May 23 19:13:20 2025

for topography visualization

@author: Giannis
"""

import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

ld = xr.open_dataset("outputs_low_resolution_model.nc")
hd = xr.open_dataset("outputs_high_resolution_model.nc")

topographic_property = "aspect"  #dem, slope, aspect
topographic_name = "Aspect"
resolutionLD = 0.1
resolutionHD = 0.01
colormap = "twilight"
label = "Aspect [°]"

topoLD = ld[str(topographic_property)]
topoHD = hd[str(topographic_property)]


#%% multiplot
degree_spacing = 0.5


print('Mapping...')

# order: t2mLD, t2mLD_pred, t2m_pred, t2m_res
# dont change the order of the above
# or change it everywhere the same below!

titles = [
    f"{topographic_name} - {resolutionLD}°",
    f"{topographic_name} - {resolutionHD}°",
    ]

mins = [
    np.nanmin(topoLD.values),
    np.nanmin(topoHD.values),
    ]
maxes = [
    np.nanmax(topoLD.values),
    np.nanmax(topoHD.values),
    ]

fig, axes = plt.subplots(
    2, 1,  #rows, columns
    figsize=(7, 10), 
    subplot_kw={'projection': ccrs.PlateCarree()}
    )

vmin = np.floor( np.min(mins) )
vmax = np.ceil( np.max(maxes) )

for ax, da, title in zip(
        axes.flat, 
        [topoLD, topoHD], 
        titles
        ):
    
    img = da.plot.pcolormesh(
        ax=ax, cmap=colormap, transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax, add_colorbar=False
        )

    ax.coastlines(resolution="10m", linewidth=0.25)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.25)
    ax.add_feature(cfeature.LAND)
    ax.set_title(title)
    
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.25, linestyle="--", color="gray",
        alpha=0.5
        )
    gl.xlocator = MultipleLocator(degree_spacing)
    gl.ylocator = MultipleLocator(degree_spacing)
    gl.top_labels = False
    gl.right_labels = False
    

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.11, 0.04, 0.78])
cbar = fig.colorbar(img, cax=cbar_ax)
cbar.set_label(label)
    
#plt.tight_layout() 
plt.savefig(f"z-{topographic_property}.png", dpi=1500, bbox_inches="tight")
plt.show() 
    