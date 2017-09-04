"""
__file__

    3_gen_distance_feat.py

__description__

    This file generates the following features

        1. distance_haversine
        2. distance_dummy_manhattan
        3. direction
        4. pca_manhattan
        5. avg_speed_h (train only)
        6. avg_speed_m (train only)

__author__

    Chenglong Chen < c.chenglong@gmail.com >
    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >
"""

import sys
import pickle
sys.path.append("../")
from param_config import config
from utils import *


def extract_distance_feat(df):
    df['distance_haversine'] = haversine_array(df['pickup_latitude'].values,
                                               df['pickup_longitude'].values,
                                               df['dropoff_latitude'].values,
                                               df['dropoff_longitude'].values)

    df['distance_dummy_manhattan'] = dummy_manhattan_distance(df['pickup_latitude'].values,
                                                              df['pickup_longitude'].values,
                                                              df['dropoff_latitude'].values,
                                                              df['dropoff_longitude'].values)

    df['direction'] = bearing_array(df['pickup_latitude'].values,
                                    df['pickup_longitude'].values,
                                    df['dropoff_latitude'].values,
                                    df['dropoff_longitude'].values)

    df['pca_manhattan'] = np.abs(df['dropoff_pca1'] - df['pickup_pca1']) + \
                          np.abs(df['dropoff_pca0'] - df['pickup_pca0'])


def extract_speed_feat(df):
    """
    for train data only
    """
    df['avg_speed_h'] = 1000 * df['distance_haversine'] / df['trip_duration']
    df['avg_speed_m'] = 1000 * df['distance_dummy_manhattan'] / df['trip_duration']


if __name__ == "__main__":
    # Load Data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)

    # Generate Features
    print "Generate diastance features...",
    extract_distance_feat(dfTrain)
    extract_distance_feat(dfTest)
    extract_speed_feat(dfTrain)

    # Dump Data
    with open(config.processed_train_data_path, "wb") as f:
        pickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        pickle.dump(dfTest, f, -1)
    print "Done."
