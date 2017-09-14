"""
__file__

    1_gen_time_feat.py

__description__

    This file generates the following features

        1. pickup_date
        2. pickup_weekday
        3. pickup_hour_weekofyear
        4. pickup_hour
        5. pickup_minute
        6. pickup_dt
        7. pickup_week_hour
        8. week_delta
        9. week_delta_sin
        10. hour_sin

__author__

    Chenglong Chen < c.chenglong@gmail.com >
    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >
"""

import sys
import pickle
import numpy as np
import datetime as dt
sys.path.append("../")
from param_config import config


def extract_time_feat(df):
    df['pickup_date']            = df['pickup_datetime'].dt.date
    df['pickup_weekday']         = df['pickup_datetime'].dt.weekday
    df['pickup_hour_weekofyear'] = df['pickup_datetime'].dt.weekofyear
    df['pickup_hour']            = df['pickup_datetime'].dt.hour
    df['pickup_minute']          = df['pickup_datetime'].dt.minute
    df['pickup_dt']              = (df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()
    df['pickup_week_hour']       = df['pickup_weekday'] * 24 + df['pickup_hour']
    df['week_delta']             = df['pickup_datetime'].dt.weekday + ((df['pickup_datetime'].dt.hour + (df['pickup_datetime'].dt.minute / 60.0)) / 24.0)
    df['week_delta_sin'] = np.sin((df['week_delta'] / 7) * np.pi)**2
    df['hour_sin'] = np.sin((df['pickup_hour'] / 24) * np.pi)**2

if __name__ == "__main__":

    # Load Data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)

    # Generate Features
    print("Generate time features...", end="")
    extract_time_feat(dfTrain)
    extract_time_feat(dfTest)

    # Dump Data
    with open(config.processed_train_data_path, "wb") as f:
        pickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        pickle.dump(dfTest, f, -1)
    print("Done.")
