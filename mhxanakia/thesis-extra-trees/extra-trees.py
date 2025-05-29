# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:03:12 2025

@author: yiann
"""

#True to train - else it imports a model
train_model = True  
save_trained_model = True

model_imported = (
    "ExtraTreesRegressor_coraviates-latitude-longitude-dem-slope-aspect-valid_month-0.1deg-trained.pkl"
    )  # set train_model = False to use this

# this year and above apply the downscaling procedure
downscaling_year = 2017
holdout_test_year = 2014

# dont include variables in the model
exclude_year = True
exclude_land_cover = False

#True for mapping - else no mapping
visualize = True

export_to_device = True


target_resolution = 0.01
coarse_resolution = 0.1

scaling_factor = coarse_resolution/target_resolution



#%% importations
print('Importing Libraries...')

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import generic_filter
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
from matplotlib.colors import ListedColormap, BoundaryNorm

 
print('Importing Data...')
dfLD = pd.read_parquet(f'df-{coarse_resolution}deg.parquet')
dfHD = pd.concat([
    pd.read_parquet(f'df-2017-2021-{target_resolution}deg.parquet'),
    pd.read_parquet(f'df-2022-2022-{target_resolution}deg.parquet')
    ])

if np.nanmean(dfLD.t2m)>200:
    dfLD["t2m"] = dfLD.t2m - 273.15
if np.nanmean(dfHD.t2m)>200:
    dfHD["t2m"] = dfHD.t2m - 273.15

dfLD_ = dfLD[dfLD.valid_year<=downscaling_year]
#dfHD = dfHD[dfHD.valid_year>=downscaling_year]

#dfLD = dfLD.drop(columns=["valid_year"])
#dfHD = dfHD.drop(columns=["valid_year"])

#dfLD = dfLD.drop(columns=["aspect"])
#dfHD = dfHD.drop(columns=["aspect"])

#dfLD = dfLD.drop(columns=["slope"])
#dfHD = dfHD.drop(columns=["slope"])


os.makedirs("images-maps", exist_ok=True)
os.makedirs("modelakia", exist_ok=True)
os.makedirs("output-nc-files", exist_ok=True)


dsLD = xr.open_dataset(f"t2m-{coarse_resolution}deg.nc")
if np.nanmean(dsLD.t2m)>200:
    dsLD["t2m"] = dsLD.t2m - 273.15

dsHD = xr.open_dataarray(f"t2m-{target_resolution}deg.nc")
dsHD = dsHD.to_dataset(name="t2mLDonHD")
dsHD = dsHD.sel(valid_time=dsHD.valid_time.dt.year >= downscaling_year)
if np.nanmean(dsHD.t2mLDonHD)>200:
    dsHD["t2mLDonHD"] = dsHD.t2mLDonHD - 273.15


#%% train/test split, import or train/optimize model
dfLD_ = dfLD_.dropna(subset=['t2m'])  # not fault-proof for all NaNs!

dfLD_train = dfLD_.loc[dfLD_['valid_year']<holdout_test_year]
dfLD_test = dfLD_.loc[dfLD_['valid_year']>=holdout_test_year]

if exclude_year == True:
    if "valid_year" in dfLD_.columns:
        dfLD_ = dfLD_.drop(columns=["valid_year"])
        dfLD_train = dfLD_train.drop(columns=["valid_year"])
        dfLD_test = dfLD_test.drop(columns=["valid_year"])
        
train_cols = list(dfLD_.columns)[1:] #exclude t2m

X_train = dfLD_train.loc[:,train_cols]
y_train = dfLD_train.loc[:,'t2m']

X_test = dfLD_test.loc[:,train_cols]
y_test = dfLD_test.loc[:,'t2m']


'''
# random split of timeseries data is not ideal...
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
'''
'''
if exclude_land_cover == True:
    if "land_cover" in dfLD_.columns:
        dfLD_ = dfLD_.drop(columns=["land_cover"])
'''      
        
# model gets trained
if train_model == True:
    
    #peloponese means that those hyperparameters were selected after
    # turning fo ra 30x30 grid containing the peloponese:
        #n_estimators=100, #peloponese
        #random_state=42, #peloponese
        #max_depth=18, #peloponese
        #min_samples_leaf=1, #peloponese
        #min_samples_split=20, #peloponese
        #max_features=None #peloponese
    # for other areas, the model might need retuning, havent tried yet...
    best_model = ExtraTreesRegressor(n_estimators=100, #100-pelop/100-optimalest/500-bayesian
                                     random_state=42, #42 pantou
                                     max_depth=24, #18-pelop/32-optimalest/24-bayesian
                                     min_samples_leaf=2, #1-pelop/2-optimalest/2-bayesian
                                     min_samples_split=50, #20-pelop/50-optimalest/50-bayesian
                                     max_features=None, #None-pelop/0.9-optimalest/0.5-bayesian
                                     bootstrap=False,
                                     #warm_start=True
                                     )
    # optimalest has slightly more rtraining accuracy, and slightly lower
    # testing accuracy
    ### no year: 18, 19, 2, None
    ### with year: None, 1, 20, None
    print(f'\nModel Used: {type(best_model).__name__}')
    print('Training with Fixed Hyperparameters...')
    best_model.fit(X_train, y_train)
    
    y_train_est = best_model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_est)
    print(f"Train set MSE of the Model: {train_mse:.4f}")
    
    
# model gets imported
else:
    print('Predicting Using Imported Model...')
    model_imported_path = os.path.join("modelakia", model_imported)
    best_model = joblib.load(model_imported_path)
    
    y_train_est = best_model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_est)
    print(f"Train set MSE of the Model: {train_mse:.4f}")

    
# GIA CV THELEI OLO TO SET - GIA KANONIKA TO TEST - DIORTHWSE TO
# make predictions on hold-out set and show some stats
y_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Hold-out Test set MSE of the Model: {test_mse:.4f}")


print('\nParameters of the Model:')
print(best_model.get_params())

features_in_model = best_model.feature_names_in_
feature_importances = best_model.feature_importances_
print("\nFeatures of the Model:", features_in_model)
print('Feature Importance:')
for i,feature in enumerate(features_in_model):
    print(f'{feature}:  {np.round(100*feature_importances[i],3)} %')


covariates = "-".join(features_in_model)
model_name = f'{type(best_model).__name__}_coraviates-{covariates}-{coarse_resolution}deg-trained.pkl'

if train_model == True:
    model_path = os.path.join("modelakia", model_name)
    if save_trained_model == True:
        joblib.dump(best_model, model_path)

#dfLD["t2m_pred"] = best_model.predict(dfLD.loc[:,train_cols])


#%% evaluation of model
y_train_df = pd.DataFrame({
    "y_train":y_train,
    "y_train_est":pd.Series(y_train_est, index=y_train.index)
        })#.sort_index()

ymin = np.min([y_train_df.y_train_est.min(), y_train_df.y_train.min()])
ymax = np.max([y_train_df.y_train_est.max(), y_train_df.y_train.max()])
x= np.linspace(np.floor(ymin), np.ceil(ymax), 100)

y_train_df["years"] = y_train_df.index.get_level_values("valid_time").year
y_train_df["latitude"] = y_train_df.index.get_level_values("latitude")
y_train_df["longitude"] = y_train_df.index.get_level_values("longitude")
'''
for year in np.unique(y_train_df.years):
    print(year)
    plt.scatter(
        y_train_df.y_train[y_train_df.years==year], 
        y_train_df.y_train_est[y_train_df.years==year], 
        s=0.05, #c=y_train_df.years, cmap="tab10"
        )
    plt.plot(x, x, linewidth=0.5, c="k")
    plt.ylabel("Estimated ERA5-Land Temperature [°C]")
    plt.xlabel("ERA5-Land Temperature [°C]")
    plt.axis("square")
    plt.grid(alpha=0.35, linestyle="--")
    plt.show()
'''
# all years together
base_cmap = plt.cm.nipy_spectral  # or 'plasma', 'turbo', etc.
unique_years = sorted(np.unique(y_train_df.years))
cmap = ListedColormap(base_cmap(np.linspace(0, 1, len(unique_years))))
norm = BoundaryNorm(boundaries=np.arange(min(unique_years), max(unique_years) + 2), ncolors=len(unique_years))

plt.scatter(
    y_train_df.y_train, 
    y_train_df.y_train_est, 
    s=0.05, alpha=0.75,
    c=y_train_df.years, 
    cmap=cmap, norm=norm
    )
plt.plot(x, x, linewidth=0.5, c="k")
plt.ylabel("Estimated ERA5-Land Temperature [°C]")
plt.xlabel("ERA5-Land Temperature [°C]")
plt.axis("square")
plt.title(f"{covariates}")
plt.grid(alpha=0.35, linestyle="--")
#plt.minorticks_on()
#plt.grid(which="minor", linestyle="--", alpha=0.35)
cbar = plt.colorbar(ticks=unique_years[::3])
cbar.set_label('Year')
#plt.savefig("model-vs-era5land-coarse-resolution.png", dpi=300, bbox_inches="tight")
plt.show()


#%% fitting the imported/optimized model t o high resolution data
print('Fitting on High Resolution Covariates...')
# better do column selection based on train_cols??
# migh tbe bette rt osplit the model.predict() in chunks. also maybe convert
# to numpy instead of dataframe? but then i wont be able to reconvert it to
# xarray? ,aybe keep the index in a separate variable, and slap it on the
# resulting prediction array?
downscale = dfHD.loc[:,train_cols]
    
downscale = best_model.predict(downscale)
dfHD['t2mHD'] = downscale
    
# to HD dataset paizei na vgazei komple pragmata kai xwris nan replacement
# isws xreiastei nan replacement gia thalasia merh mono
dfHD['t2mHD'] = dfHD['t2mHD'].where(dfHD['t2m'].notna()) #vazw NaN values


#dsHD = dfHD.to_xarray()
valid_timesHD = dfHD.index.get_level_values('valid_time').unique()

# since i have the multiindex, it can be safely done through .to_xarray()
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

res['residual'] =  res.t2m - res.t2m_pred  #edw eisai
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

# idio apotelesma me thn ftiaxtikh
def fill_adjacent_nans(arr):
    arr = arr.copy()

    def nan_mean_filter(values):
        center = values[len(values) // 2]
        if np.isnan(center):
            neighbors = np.array(values)
            # Exclude the center value and NaNs from the mean
            neighbors = neighbors[~np.isnan(neighbors)]
            if len(neighbors) > 0:
                return np.mean(neighbors)
        return center

    kernel_size = 3
    # Apply filter iteratively to fill only adjacent NaNs
    while True:
        arr_before = arr.copy()
        arr = generic_filter(arr, nan_mean_filter, size=kernel_size, mode='constant', cval=np.nan)
        
        if np.array_equal(np.nan_to_num(arr), np.nan_to_num(arr_before)):
            break
        
    return arr

def fill_adjacent_nans_across_depth(data_3d):
    filled = data_3d.copy()
    for i in range(data_3d.shape[0]):  # Loop over "depth" axis
        filled[i] = fill_adjacent_nans(data_3d[i])
    return filled


print("Extracting High Resolution Residuals...")
print("Using Bilinear Interpolation...")
latHD_shape = dsHD.latitude.shape[0]
lonHD_shape = dsHD.longitude.shape[0]

# fill NaNs with 0
resLD_filled = np.nan_to_num(resLD.resLD.to_numpy(), nan=0)
# fill NaNs with the closest neighbours
resLD_filled1 = fill_adjacent_nans_3d(resLD.resLD.to_numpy())
resLD_filled2 = fill_adjacent_nans_with_edges(resLD.resLD.to_numpy())
# auto dinei to idio apotelesma me thn ftiaxtikh mou...
#resLD_filled3 = fill_adjacent_nans_across_depth(resLD.resLD.to_numpy())


resHD = np.array([
    resize(img, (latHD_shape, lonHD_shape), order=1, anti_aliasing=True) 
    for img in resLD_filled2
    ])


resHD[np.isnan(dsHD.t2mLDonHD.values)] = np.nan
#resHD0[np.isnan(dsHD.t2mLDonHD.values)] = np.nan

resHD = xr.DataArray(resHD, dims=dsHD.dims, coords=dsHD.coords)
#resHD0 = xr.DataArray(resHD0, dims=dsHD.dims, coords=dsHD.coords)

dsHD["resHD"] = resHD
#dsHD["resHD0"] = resHD0  # athliooo

print("Done.")


#%% export to device
if export_to_device == True:
    resLD.to_netcdf(
        os.path.join("output-nc-files", f"outputs-{coarse_resolution}deg.nc")
        )
    dsHD.to_netcdf(
        os.path.join("output-nc-files", f"outputs-{target_resolution}deg.nc")
        )


#%% optikopoihsh
degree_spacing = 0.2
temporal_idx = 47

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
        figsize=(17, 12), 
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

        ax.coastlines(resolution="10m", linewidth=0.2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.2)
        ax.add_feature(cfeature.LAND)
        ax.set_title(title)
        
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.2, linestyle="--", color="gray"
            )
        gl.xlocator = MultipleLocator(0.4)
        gl.ylocator = MultipleLocator(0.4)
        gl.top_labels = False
        gl.right_labels = False
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.04, 0.78])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label("Temperature [°C]")
    
    #plt.tight_layout()  
    plt.savefig(f"multiplot{temporal_idx}-{target_resolution}deg.png", dpi=1000, bbox_inches="tight")
    plt.show() 
    
    
