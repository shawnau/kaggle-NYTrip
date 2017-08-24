"""
__file__

    6_gen_osrm_feat.py

__description__

    This file generates the following features

        1. total_distance
        2. total_travel_time
        3. number_of_steps

__author__

    Chenglong Chen < c.chenglong@gmail.com >
    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >
"""

import sys
import pickle
import pandas as pd
sys.path.append("../")
from param_config import config


def extract_osrm_feat(train, test):
    cols = ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
    fr1 = pd.read_csv('%s/fastest_routes_train_part_1.csv' % config.data_folder, usecols=cols)
    fr2 = pd.read_csv('%s/fastest_routes_train_part_2.csv' % config.data_folder, usecols=cols)

    train_street_info = pd.concat((fr1, fr2))
    test_street_info = pd.read_csv('%s/fastest_routes_test.csv' % config.data_folder, usecols=cols)

    train = train.merge(train_street_info, how='left', on='id')
    test = test.merge(test_street_info, how='left', on='id')

    return train, test

if __name__ == "__main__":
    # Load Data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)

    # Generate Features
    print("Generate osrm features...", end="")
    dfTrain, dfTest = extract_osrm_feat(dfTrain, dfTest)

    # Dump Data
    with open(config.processed_train_data_path, "wb") as f:
        pickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        pickle.dump(dfTest, f, -1)
    print("Done.")
