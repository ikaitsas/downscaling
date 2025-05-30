# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:03:12 2025

@author: yiann

Moving or expanding window folds should be tried...
"""
downscaling_year = 2017 

n_total_iters = 500
n_per_chunk = 100

checkpoint_path = "bayes_checkpoint.pkl"
best_models = []
best_scores = []


#%% importations
print('Importing Libraries...')

import os
import time
import numpy as np
import pandas as pd

os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras  #needed for NNs

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import joblib

from skopt.callbacks import CheckpointSaver
from skopt import dump, load


def safe_mse(y_true, y_pred):
    # Some parameter combinations may retur NaN y_pred values?
    # This error function allows the optimization to run
    # despite the occasional production of those NaNs, so that
    # an especially long optimization process continues running
    # and completes robustly
    # ideally track y_pred values, cause these extreme min/max
    # values might cause the hyperparameter optimization 
    # process to crash...
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return 9999  # very bad score
    if np.any((y_pred < -500) | (y_pred > 500)):  # overflow or unreasonable values
        return 9999
    return mean_squared_error(y_true, y_pred)


print('Importing Data...')
dfLD = pd.read_parquet('df.parquet')
#dfHD = pd.read_parquet('dfHD.parquet')

dfLD = dfLD[dfLD.valid_year<downscaling_year]
#dfHD = dfHD[dfHD.valid_year>=2020]

dfLD = dfLD.drop(columns=["valid_year"])
#dfHD = dfHD.drop(columns=["valid_year"])


if np.nanmean(dfLD.t2m)>200:
    dfLD["t2m"] = dfLD.t2m - 273.15


#%% train/test split, import or train/optimize model
dfLD_ = dfLD.dropna(subset=['t2m'])
train_cols = list(dfLD_.columns)[1:] #exclude t2m
X = dfLD_.loc[:,train_cols] 
y = dfLD_.loc[:,'t2m']

# might be better to perform a train-test split non-randomly
# in order to capitalize the temporal consistency of the data?
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



#%% optimization parameters gets setup  
model = ExtraTreesRegressor(random_state=42)
print(f'\nModel Used: {type(model).__name__}')
print('Setting up search space, CV, scorer...')

search_space = {
    "n_estimators": Integer(6, 500),  
    "max_depth": Categorical([None] + list(range(4, 64, 4))),  
    "min_samples_split": Integer(2, 50),  
    "min_samples_leaf": Integer(2, 100),  
    "max_features": Categorical(
        ['sqrt', 'log2', None] + list(
            np.round(np.arange(0.2,1.1,0.1),2)
            )
        ) 
}

kfold = KFold(n_splits=10, shuffle=False, random_state=None)

scorer = make_scorer(safe_mse, greater_is_better=False)


#%% optimization loop
print("Performing BayesSearchCV...")
if os.path.exists(checkpoint_path):
    print("Resuming from checkpoint...")
    bayes_search = load(checkpoint_path)
    completed_iters = len(bayes_search.cv_results_['params'])
else:
    completed_iters = 0


os.makedirs("optimization-stuff", exist_ok=True)


# Chunked loop
while completed_iters < n_total_iters:
    remaining = n_total_iters - completed_iters
    run_now = min(n_per_chunk, remaining)
    print(f"Running {run_now} more iterations...")

    model = ExtraTreesRegressor(random_state=42)
    bayes_search = BayesSearchCV(
        model,
        search_space,
        n_iter=run_now,
        cv=kfold,
        scoring=scorer,
        n_jobs=1,
        verbose=3,
        random_state=42,
        refit=True
    )

    checkpoint_cb = CheckpointSaver(checkpoint_path, compress=True, store_objective=False)
    bayes_search.fit(X_train, y_train, callback=[checkpoint_cb])

    # Save best model of this chunk
    chunk_best_path = os.path.join(
        "optimization-stuff", f"best_model_iter_{completed_iters+run_now}.pkl"
        )
    joblib.dump(bayes_search.best_estimator_, chunk_best_path)
    best_models.append(
        (bayes_search.best_score_, 
         bayes_search.best_params_,
         bayes_search.best_estimator_
         )
        )
    best_scores.append(bayes_search.best_score_)
    
    output_text_path = os.path.join(
        "optimization-stuff", f"text_best_model_iter_{completed_iters+run_now}.txt"
        )
    with open(output_text_path, "a") as f:
        f.write(f"Iteration {completed_iters+run_now}\n")
        f.write(f"Score: {bayes_search.best_score_}\n")
        f.write(f"Params: {bayes_search.best_params_}\n")
        f.write(f"Params: {bayes_search.best_estimator_}\n\n\n")
    
    completed_iters += run_now
    print(f"\nCompleted: {completed_iters}/{n_total_iters}\n")
    

with open("best_models.txt", "w") as f:
    for best_model in best_models:
        f.write(str(best_model))
        f.write("\n\n\n")



#%% without loop
'''
bayes_search = BayesSearchCV(
    model,
    search_space,
    n_iter=750,  
    cv=kfold,  
    scoring=scorer,
    n_jobs=1,  # -1 for all CPU cores
    verbose=3,
    random_state=42
)
# verbose only works for n_jobs=1 (others might work on linux?)
# will probably add a logging setup to track progress

bayes_search.fit(X_train, y_train)


print("\nBest parameters found by BayesSearchCV:")
print(bayes_search.best_params_)
print("\nBest cross-validated score (negative MSE):")
print(bayes_search.best_score_)
'''
'''
print('\nTraining Best Model on the Whole Dataset...')

best_model = bayes_search.best_estimator_
best_model.fit(X, y)
'''

#%% sxolia
'''
# GIA CV THELEI TA TRAIN-VAL SETS - GIA KANONIKA TO TEST

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


