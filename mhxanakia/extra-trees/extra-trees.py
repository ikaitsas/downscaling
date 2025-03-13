# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:03:12 2025

@author: yiann
"""

#True to train - else it imports a model
train_model = True  
save_trained_model = False

model_imported = (
    'ExtraTreesRegressor_coraviates-'
    'latitude-longitude-dem-slope-valid_year-valid_month.pkl'
    )

#True for mapping - else no mapping
visualize = True  


#%% importations
print('Importing Libraries...')

import os
import numpy as np
import pandas as pd
import xarray as xr
from skimage.transform import resize

os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras  #needed for NNs

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator


# this year and above apply the downscaling procedure
downscaling_year = 2020

# dont include year in the model
exclude_year = True

 
print('Importing Data...')
dfLD = pd.read_parquet('df.parquet')
dfHD = pd.read_parquet('dfHD.parquet')

dfLD["t2m"] = dfLD.t2m - 273.15
dfHD["t2m"] = dfHD.t2m - 273.15

dfLD_ = dfLD[dfLD.valid_year<downscaling_year]
dfHD = dfHD[dfHD.valid_year>=downscaling_year]

#dfLD = dfLD.drop(columns=["valid_year"])
#dfHD = dfHD.drop(columns=["valid_year"])

#dfLD = dfLD.drop(columns=["aspect"])
#dfHD = dfHD.drop(columns=["aspect"])

#dfLD = dfLD.drop(columns=["slope"])
#dfHD = dfHD.drop(columns=["slope"])


os.makedirs("images-maps", exist_ok=True)
os.makedirs("modelakia", exist_ok=True)


dsLD = xr.open_dataset("t2m.nc")
dsLD["t2m"] = dsLD.t2m - 273.15

dsHD = xr.open_dataarray("t2mHD.nc")
dsHD = dsHD.to_dataset(name="t2mLDonHD")
dsHD = dsHD.sel(valid_time=dsHD.valid_time.dt.year >= downscaling_year)
dsHD["t2mLDonHD"] = dsHD.t2mLDonHD - 273.15



#%% train/test split, import or train/optimize model
dfLD_ = dfLD_.dropna(subset=['t2m'])
if exclude_year == True:
    dfLD_ = dfLD_.drop(columns=["valid_year"])
train_cols = list(dfLD_.columns)[1:] #exclude t2m
X = dfLD_.loc[:,train_cols] 
y = dfLD_.loc[:,'t2m']

'''
X_train = dfLD_[dfLD_.valid_year>1996].loc[:,train_cols]
X_test = dfLD_[dfLD_.valid_year<=1996].loc[:,train_cols]

y_train = dfLD_[dfLD_.valid_year>1996].loc[:,"t2m"]
y_test = dfLD_[dfLD_.valid_year<=1996].loc[:,"t2m"]
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# model gets trained
if train_model == True:
        
    best_model = ExtraTreesRegressor(n_estimators=100, 
                                     random_state=42, 
                                     max_depth=18, #
                                     min_samples_leaf=1, #
                                     min_samples_split=20, #
                                     max_features=None #
                                     )
    # no year: 18, 19, 2, None
    # with year: None, 1, 20, None
    print(f'\nModel Used: {type(best_model).__name__}')
    print('Training with Fixed Hyperparameters...')
    best_model.fit(X_train, y_train)
    
    

# model gets imported
else:
    print('Predicting Using Imported Model...')
    model_imported_path = os.path.join("modelakia", model_imported)
    best_model = joblib.load(model_imported_path)

    
    
# GIA CV THELEI OLO TO SET - GIA KANONIKA TO TEST - DIORTHWSE TO
# make predictions on hold-out set and show some stats
y_pred = best_model.predict(X_test)
random_mse = mean_squared_error(y_test, y_pred)
print(f"Hold-out Test set MSE of the Model: {random_mse:.4f}")

print('\nParameters of the Model:')
print(best_model.get_params())

features_in_model = best_model.feature_names_in_
feature_importances = best_model.feature_importances_
print("\nFeatures of the Model:", features_in_model)
print('Feature Importance:')
for i,feature in enumerate(features_in_model):
    print(f'{feature}:  {np.round(100*feature_importances[i],3)} %')


covariates = "-".join(features_in_model)
model_name = f'{type(best_model).__name__}_coraviates-{covariates}.pkl'

if train_model == True:
    model_path = os.path.join("modelakia", model_name)
    if save_trained_model == True:
        joblib.dump(best_model, model_path)

#dfLD["t2m_pred"] = best_model.predict(dfLD.loc[:,train_cols])


#%% fitting the imported/optimized model t o high resolution data
print('Fitting on High Resolution Covariates...')
#better do column selection based on train_cols??
'''
if 't2mHD' in dfHD.columns:
    downscale = dfHD.drop(['t2m', 't2mHD'], axis=1)
                                                 
else:
    downscale = dfHD.drop(['t2m'], axis=1)

if exclude_year == True:
    downscale = downscale.drop(["valid_year"], axis=1)
'''
downscale = dfHD.loc[:,train_cols]
    
downscale = best_model.predict(downscale)
dfHD['t2mHD'] = downscale
    
#to HD dataset paizei na vgazei komple pragmata kai xwris nan replacement
#isws xreiastei nan replacement gia thalasia merh mono
dfHD['t2mHD'] = dfHD['t2mHD'].where(dfHD['t2m'].notna()) #vazw NaN values
    

#dsHD = dfHD.to_xarray()
valid_timesHD = dfHD.index.get_level_values('valid_time').unique()

t2mHD = xr.DataArray(
    dfHD.t2mHD.values.reshape(
        len(valid_timesHD), 
        len(dsHD.latitude.data), 
        len(dsHD.longitude.data)
        ),
    coords={
        'valid_time': valid_timesHD, 
        'latitude': dsHD.latitude.data, 
        'longitude': dsHD.longitude.data
        },
    dims=['valid_time', 'latitude', 'longitude'],
    name='t2mHD'
)

dsHD["t2mHD"] = t2mHD
  
print("Done.")


#%% residual fitting
# res contains coarse resolution data corresponding to
# downscaling_year, to extract the coarse resolution 
# residuals, and downscale them by bilinear interpolation
print("Fitting Residuals on Coarse Data...")
res = dfLD.loc[dfLD.valid_year>=downscaling_year,:].copy()

res["t2m_pred"] = best_model.predict(res.loc[:,train_cols])
res['t2m_pred'] = res['t2m_pred'].where(res['t2m'].notna())

res['residual'] =  res.t2m - res.t2m_pred
res['residual'] = res['residual'].where(res['t2m'].notna())

valid_times = res.index.get_level_values('valid_time').unique()

t2m_predLD = xr.DataArray(
    res.t2m_pred.values.reshape(
        len(valid_times), 
        len(dsLD.latitude.values), 
        len(dsLD.longitude.values)
        ),
    coords={
        'valid_time': valid_times, 
        'latitude': dsLD.latitude.values, 
        'longitude': dsLD.longitude.values
        },
    dims=['valid_time', 'latitude', 'longitude'],
    name='t2m_predLD'
)

residualLD = xr.DataArray(
    res.residual.values.reshape(
        len(valid_times), 
        len(dsLD.latitude.values), 
        len(dsLD.longitude.values)
        ),
    coords={
        'valid_time': valid_times, 
        'latitude': dsLD.latitude.values, 
        'longitude': dsLD.longitude.values
        },
    dims=['valid_time', 'latitude', 'longitude'],
    name='resLD'
)

resLD = dsLD.sel(valid_time=dsLD.valid_time.dt.year >= downscaling_year)
resLD["t2m_predLD"] = t2m_predLD
resLD["resLD"] = residualLD


# produce HD version of residuals, using bilinear interpolation

# the following function needs to only replace NaNs adjucent to
# non-NaNs, not the entire NaN population
# will modify accordingly in the future...
# also need to take into account the "edge" cells
def fill_adjacent_nans_3d(grid):
    """
    Fills NaN values in a 3D NumPy array along axes 1 and 2 (rows and columns).
    - If a NaN cell has only one non-NaN neighbor, use nearest neighbor.
    - If a NaN cell has multiple non-NaN neighbors, use the average.

    Parameters:
    grid (np.ndarray): 3D NumPy array with NaNs

    Returns:
    np.ndarray: Filled grid
    """
    grid = grid.copy()  # Avoid modifying the original array
    depth, rows, cols = grid.shape

    # 4-adjacency (left, right, up, down)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for d in range(depth):  # Process each slice separately
        nan_positions = np.argwhere(np.isnan(grid[d]))  # Find NaN positions

        for r, c in nan_positions:
            neighbors = []

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not np.isnan(grid[d, nr, nc]):
                    neighbors.append(grid[d, nr, nc])

            if len(neighbors) == 1:  # Nearest neighbor interpolation
                grid[d, r, c] = neighbors[0]
            elif len(neighbors) > 1:  # Average of multiple neighbors
                grid[d, r, c] = np.mean(neighbors)

    return grid

def fill_adjacent_nans_with_edges(grid):
    """
    Fills NaN values in a 3D NumPy array along axes 1 and 2 (rows and columns).
    - If a NaN cell has only one adjacent non-NaN, use that value (nearest neighbor).
    - If a NaN cell has multiple adjacent non-NaNs, take the average.
    - Expands the search to include "edge" neighbors.

    Parameters:
    grid (np.ndarray): 3D NumPy array with NaNs

    Returns:
    np.ndarray: Filled grid
    """
    grid = grid.copy()  # Avoid modifying the original array
    depth, rows, cols = grid.shape

    # 4-adjacency (left, right, up, down)
    primary_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # "Edge" adjacency (diagonal-like but considering only common adjacent cells)
    edge_dirs = [
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal neighbors
    ]

    changed = True  # Flag to track if we need another pass

    while changed:
        changed = False  # Reset flag
        new_grid = grid.copy()  # Work on a copy to avoid overwriting in the loop

        for d in range(depth):  # Process each slice independently
            nan_positions = np.argwhere(np.isnan(grid[d]))  # Find NaN positions

            for r, c in nan_positions:
                neighbors = []

                # Check primary (directly adjacent) neighbors
                for dr, dc in primary_dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not np.isnan(grid[d, nr, nc]):
                        neighbors.append(grid[d, nr, nc])

                # Check edge (diagonal-like) neighbors
                for dr, dc in edge_dirs:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows and 0 <= nc < cols and
                        not np.isnan(grid[d, nr, nc])
                    ):
                        neighbors.append(grid[d, nr, nc])

                # Fill NaN if there are valid non-NaN neighbors
                if neighbors:
                    new_value = np.mean(neighbors) if len(neighbors) > 1 else neighbors[0]
                    new_grid[d, r, c] = new_value
                    changed = True  # Keep iterating if we modified at least one cell

        grid = new_grid  # Update grid with new values

    return grid


print("Extracting High Resolution Residuals...")
print("Using Bilinear Interpolation...")
latHD_shape = dsHD.latitude.shape[0]
lonHD_shape = dsHD.longitude.shape[0]

# fill NaNs with 0
resLD_filled = np.nan_to_num(resLD.resLD.to_numpy(), nan=0)  
# fill NaNs with the closest neighbours
resLD_filled1 = fill_adjacent_nans_3d(resLD.resLD.to_numpy())
resLD_filled2 = fill_adjacent_nans_with_edges(resLD.resLD.to_numpy())


resHD = np.array([
    resize(img, (latHD_shape, lonHD_shape), order=1, anti_aliasing=True) 
    for img in resLD_filled2
    ])

resHD[np.isnan(dsHD.t2mLDonHD.values)] = np.nan

resHD = xr.DataArray(resHD, dims=dsHD.dims, coords=dsHD.coords)
dsHD["resHD"] = resHD

print("Done.")


#%% optikopoihsh
degree_spacing = 0.1
temporal_idx = 11

if visualize == True:
    
    t2m_pred = dsHD.t2mHD 
    t2m_res = dsHD.t2mHD + dsHD.resHD 
    t2mLD = resLD.t2m 
    t2mLD_pred = resLD.t2m_predLD
    times = dsHD.valid_time.values
    
    print('Mapping...')

    # order: t2mLD, t2mLD_pred, t2m_pred, t2m_res
    # dont change the order of the above
    # or change it everywhere the same below!
    titles = [
        (f"Low Resolution Predicted T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         ),
        (f"ERA5-Land T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         ),
        (f"Downscaled T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         ),
        (f"Residual Corrected Downscaled T2m "
         f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
         )
        ]
    
    mins = [
        np.nanmin(t2mLD_pred.values[temporal_idx,:,:]),
        np.nanmin(t2mLD.values[temporal_idx,:,:]),
        np.nanmin(t2m_pred.values[temporal_idx,:,:]),
        np.nanmin(t2m_res.values[temporal_idx,:,:])
        ]
    maxes = [
        np.nanmax(t2mLD_pred.values[temporal_idx,:,:]),
        np.nanmax(t2mLD.values[temporal_idx,:,:]),
        np.nanmax(t2m_pred.values[temporal_idx,:,:]),
        np.nanmax(t2m_res.values[temporal_idx,:,:])
        ]
    
    fig, axes = plt.subplots(
        2, 2, 
        figsize=(12, 12), 
        subplot_kw={'projection': ccrs.PlateCarree()}
        )
    
    vmin = np.floor(np.min(mins))
    vmax = np.ceil(np.max(maxes))
    
    for ax, da, title in zip(
            axes.flat, 
            [t2mLD_pred,t2mLD,t2m_pred,t2m_res], 
            titles
            ):
        
        img = da.isel(valid_time=temporal_idx).plot.pcolormesh(
            ax=ax, cmap='inferno', transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, add_colorbar=False
            )

        ax.coastlines()
        ax.set_title(title)
        
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, linestyle="--", color="gray"
            )
        gl.xlocator = MultipleLocator(0.1)
        gl.ylocator = MultipleLocator(0.1)
        gl.top_labels = False
        gl.right_labels = False
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.04, 0.78])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("Temperature [°C]")
    
    #plt.tight_layout()  
    plt.savefig("multiplot.png", dpi=1000, bbox_inches="tight")
    plt.show() 
    
    
