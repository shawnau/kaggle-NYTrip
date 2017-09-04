"""
__file__

    0_preprocess.py

__description__

    This file pre-processes dataset

__author__

    Chenglong Chen < c.chenglong@gmail.com >
    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >
"""

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../")
from param_config import config


# Load Data
print("Load data...", end="")

dfTrain = pd.read_csv(config.original_train_data_path)
dfTest = pd.read_csv(config.original_test_data_path)

print("Done.")


# Pre-process Data
print("Pre-process data...", end="")

# encode Y/N into 1/0
dfTrain['store_and_fwd_flag'] = 1 * (dfTrain.store_and_fwd_flag.values == 'Y')
dfTest['store_and_fwd_flag'] = 1 * (dfTest.store_and_fwd_flag.values == 'Y')

# reformat datetime
dfTrain['pickup_datetime'] = pd.to_datetime(dfTrain.pickup_datetime)
dfTest['pickup_datetime'] = pd.to_datetime(dfTest.pickup_datetime)

# log transform of the label to predict (turn RMSLE into RMSE)
dfTrain['log_trip_duration'] = np.log(dfTrain['trip_duration'].values + 1)

print("Done.")


# Save Data
print("Save data...", end="")

with open(config.processed_train_data_path, "wb") as f:
    pickle.dump(dfTrain, f, -1)
with open(config.processed_test_data_path, "wb") as f:
    pickle.dump(dfTest, f, -1)

print("Done.")
