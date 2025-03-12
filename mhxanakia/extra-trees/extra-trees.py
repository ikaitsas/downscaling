# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:03:12 2025

@author: yiann
"""

#True to train - else it imports a model
train_model = True  

model_imported = 'bestMhxanaki_ExtraTreesRegressor_RandomCV6.pkl'

#True for mapping - else no mapping
visualize = True  


#%% importations
print('Importing Libraries...')

import os
import numpy as np
import pandas as pd
import xarray as xr
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt

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
    
    
'''
# model gets imported
else:
    print('Predicting Using Imported Model...')
    best_model = joblib.load(model_imported)
'''
    
    
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
#joblib.dump(best_model, model_name)

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


#%% peiramata
res = dfLD.loc[dfLD.valid_year>=2020,:].copy()

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


latHD_shape = dsHD.latitude.shape[0]
lonHD_shape = dsHD.longitude.shape[0]

# fill NaNs with 0
resLD_filled = np.nan_to_num(resLD.residual.to_numpy(), nan=0)  
# fill NaNs with the closest neighbours
resLD_filled1 = fill_adjacent_nans_3d(resLD.residual.to_numpy())
resLD_filled2 = fill_adjacent_nans_with_edges(resLD.residual.to_numpy())


resHD = np.array([
    resize(img, (latHD_shape, lonHD_shape), order=1, anti_aliasing=True) 
    for img in resLD_filled2
    ])

resHD[np.isnan(dsHD.t2mLDonHD.values)] = np.nan

resHD = xr.DataArray(resHD, dims=dsHD.dims, coords=dsHD.coords)
dsHD["resHD"] = resHD


#%% optikopoihsh
degree_spacing = 0.1
temporal_idx = 11
if visualize == True:
    temp_pred = dsHD.t2mHD 
    temp_res = dsHD.t2mHD + dsHD.resHD 
    tempLD = resLD.t2m 
    tempLD_pred = resLD.t2m_predLD
    times = dsHD.valid_time.values
    print('Mapping...')

    fig, ax = plt.subplots(
        #nrows=1, ncols=2,
        subplot_kw={"projection": ccrs.PlateCarree()}
        )
    
    ax.add_feature(cfeature.LAKES.with_scale("50m"), 
                   edgecolor="black")
    
    #try the following with plt.pcolormesh() too
    #it just needs longitude, latitude, 2d-array
    #xr.plot assumes the center of the grid cell
    #plt.imshow assumes the top-left corner
    temp_res.isel(valid_time=temporal_idx).plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        cmap=plt.cm.inferno, 
        cbar_kwargs={"label": "Temperature (°C)"},
        vmin=-1, vmax=15
    )
    
    ax.coastlines(resolution="10m", linewidth=0.75)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND)
    
    gl = ax.gridlines(draw_labels=True, linestyle=":", 
                      linewidth=0.5, color="k",
                      xlocs=np.arange(
                          dfLD.longitude.min(), 
                          dfLD.longitude.max(), 
                          degree_spacing
                          ),  #or: mticker.FixedLocator
                      ylocs=np.arange(
                          dfLD.latitude.max(), 
                          dfLD.latitude.min(), 
                          -degree_spacing
                          ) 
                      )
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlabel_values = ds.longitude.values[::5]
    #gl.ylabel_values = ds.latitude.values[::5]
    '''
    gl.xformatter = mticker.FuncFormatter(
        lambda x, _: f"{x:.1f}" if x in ds.longitude.values[::5] else ""
        )
    gl.yformatter = mticker.FuncFormatter(
        lambda y, _: f"{y:.1f}" if y in ds.latitude.values[::5] else ""
        )
    '''
    #'''
    ax.set_title(
    (f"Downscaled Temperature "
     f"{np.datetime_as_string(times[temporal_idx], unit='M')}"
     )
        )
    #'''
    #ax.set_title('Corrected Downscaled Temperature - 2020-12')
    '''
    plt.savefig(
        os.path.join(
            "images-maps", 
            f't2m-era5-land-{model_name}.png'
            ),
        dpi=1000
        )
    '''
    plt.show()
    
    #t2m = None




#%% sxolia
'''
# model is additionally optimized
if optimize_model == True:
    model = ExtraTreesRegressor(random_state=42)
    print(f'\nModel Used: {type(model).__name__}')
    print('Starting Hyperparameter Optimization...')
    time_start = time.time()
    
    param_grid = {
        'n_estimators': [25, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 10, 20],
        'max_features': ['sqrt', 'log2', None, 0.3]
    }
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Performing RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid, #param_distributions for RandomCV
        n_iter=50,  # Number of parameter settings sampled
        cv=kfold,
        scoring='neg_mean_squared_error',  #might also try MAE
        verbose=3,
        random_state=42, #enable for RandomCV
        n_jobs=1
    )
    #mallon tha kanw bayesian optimization sto mellon...
    #gia kfold=5/n_iter=3 thelei 180sec konta...
    #an einai analogo, thelw polla parapanw gia exhaustive search...
    #mporei na dokimasw bayesian optimization kapoia stigmh...
    #verbose only works for n_jobs=1 (others might work on linux?)
    
    random_search.fit(X_train, y_train)
    
    time_end = time.time()
    print(f"Hyperparameter Optimization Runtime: {time_end-time_start:.2f} s")
    
    print("\nBest parameters found by RandomizedSearchCV:")
    print(random_search.best_params_)
    print("\nBest cross-validated score (negative MSE):")
    print(random_search.best_score_)
    
    print('\nTraining Best Model on the Whole Dataset...')
    
    best_model = random_search.best_estimator_
    best_model.fit(X, y)
'''


