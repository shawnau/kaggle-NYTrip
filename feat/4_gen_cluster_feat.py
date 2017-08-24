"""
__file__

    4_gen_cluster_feat.py

__description__

    This file generates the following features

        1. pickup_cluster
        2. dropoff_cluster

__author__

    Chenglong Chen < c.chenglong@gmail.com >
    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >
"""

import sys
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
sys.path.append("../")
from param_config import config


def extract_cluster_feat(model, df):
    df['pickup_cluster'] = model.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = model.predict(df[['dropoff_latitude', 'dropoff_longitude']])


if __name__ == "__main__":
    # Load Data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)

    # Generate Features
    # train kmeans
    print("Training kmeans...")
    coords = np.vstack((dfTrain[['pickup_latitude', 'pickup_longitude']].values,
                       dfTrain[['dropoff_latitude', 'dropoff_longitude']].values,
                       dfTest[['pickup_latitude', 'pickup_longitude']].values,
                       dfTest[['dropoff_latitude', 'dropoff_longitude']].values))

    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

    print("Generate cluster features...", end="")
    extract_cluster_feat(kmeans, dfTrain)
    extract_cluster_feat(kmeans, dfTest)

    # Dump Data
    with open(config.processed_train_data_path, "wb") as f:
        pickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        pickle.dump(dfTest, f, -1)
    print("Done.")
