# coding:utf-8
"""
__file__

    2_gen_coord_feat.py

__description__

    This file generates the following features

        1. pickup_pca0
        2. pickup_pca1
        3. dropoff_pca0
        4. dropoff_pca1
        5. center_latitude
        6. center_longitude
        7. pickup_lat_bin
        8. pickup_long_bin
        9. pickup_dt_bin

__author__

    Chenglong Chen < c.chenglong@gmail.com >
    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >
"""

import sys
import pickle
import numpy as np
from sklearn.decomposition import PCA
sys.path.append("../")
from param_config import config


def extrect_coord_feat(pca, df):
    df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]
    df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]
    df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    # 把路径中点坐标也作为特征
    df['center_latitude'] = (df['pickup_latitude'].values + df['dropoff_latitude'].values) / 2
    df['center_longitude'] = (df['pickup_longitude'].values + df['dropoff_longitude'].values) / 2

    # 按坐标块分组
    df['pickup_lat_bin'] = np.round(df['pickup_latitude'], 2)
    df['pickup_long_bin'] = np.round(df['pickup_longitude'], 2)
    df['center_lat_bin'] = np.round(df['center_latitude'], 2)
    df['center_long_bin'] = np.round(df['center_longitude'], 2)
    df['pickup_dt_bin'] = (df['pickup_dt'] // (3 * 3600))


if __name__ == "__main__":
    # Load Data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)

    # Generate Features
    # train PCA
    print "Training PCA..."
    coords = np.vstack((dfTrain[['pickup_latitude', 'pickup_longitude']].values,
                       dfTrain[['dropoff_latitude', 'dropoff_longitude']].values,
                       dfTest[['pickup_latitude', 'pickup_longitude']].values,
                       dfTest[['dropoff_latitude', 'dropoff_longitude']].values))
    pca = PCA().fit(coords)

    print "Generate coord features...",
    extrect_coord_feat(pca, dfTrain)
    extrect_coord_feat(pca, dfTest)

    # Dump Data
    with open(config.processed_train_data_path, "wb") as f:
        pickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        pickle.dump(dfTest, f, -1)
    print "Done."
