# train_spline_model.py

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import random
import joblib  # for saving models
import statsmodels.api as sm
from scipy.stats import t
import scipy.stats as stats
from itertools import combinations
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# Load your dataset
df = pd.read_csv("../Data/popcorn_pops_custom.csv")
seed = 42
df['pop_time_seconds'] = df['pop_time_seconds'].apply(lambda x: np.random.normal(150,20) if x < 20 else x) # Replace popping times <20 with > 120
df['trial'] = df['trial'] + 1

random.seed(42)

# 3:1 train/test split 
sample = random.sample(range(1, max(df['trial'])), 30)
train = pd.DataFrame()
for trial in sample:
    trial = df[df['trial'] == trial]
    train = pd.concat([train,trial])

test = pd.DataFrame()
for i in range(max(df['trial'])): 
    if i not in sample:
        trial = df[df['trial'] == i ]
        test = pd.concat([test,trial])

# Create time array and pop CDFs
time = np.arange(0, 180.01, 0.01)
pops = np.zeros((len(sample), len(time)))

for i, trial in enumerate(sample):
    pop_times = df[df['trial'] == trial]['pop_time_seconds'].values
    # Sort pop times for cumulative counting
    pop_times = np.sort(pop_times)

    # Count how many pop times are ≤ each time point
    pops[i] = np.searchsorted(pop_times, time, side='right')


# non-parametric regression on cumulative counts using one smoothing spline
# Hyperparameter tuning on smoothness s across each trial's CDF 
n_trials, n_timepoints = pops.shape
s_grid = np.logspace(-1, 4, 20)  # try a range of smoothing factors
errors = []

for s in tqdm(s_grid):
    fold_mse = []

    for i in range(n_trials):
        # Leave-one-out split
        train_cdfs = np.delete(pops, i, axis=0)
        val_cdf = pops[i]

        # Fit spline on mean of training trials
        y_train_mean = np.mean(train_cdfs, axis=0)
        spline = UnivariateSpline(time, y_train_mean, s=s)

        # Evaluate on validation trial
        y_pred = spline(time)
        mse = mean_squared_error(val_cdf, y_pred)
        fold_mse.append(mse)

    errors.append(np.mean(fold_mse))

best_s = s_grid[np.argmin(errors)]

# training cross-validated model on all of training set with best_s
final_y_train = np.mean(pops, axis=0)
final_spline = UnivariateSpline(time, final_y_train, s=best_s)

# Save the model
joblib.dump(final_spline, "spline_model.pkl")
