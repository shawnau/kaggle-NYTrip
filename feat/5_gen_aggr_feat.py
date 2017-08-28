"""
__file__

    5_gen_aggr_feat.py

__description__

    This file generates the following features

        1. [avg_speed_h, avg_speed_m, log_trip_duration]  group by [gby_cols], 6*3=18 features
        2. [avg_speed_m, count(id)] group by pairs, 5*2=10 features
        3. pickup_datetime_group
        4. count_60min:  samples in the recent 60 mins
        5. dropoff_cluster_count
        6. pickup_cluster_count

__author__

    Chenglong Chen < c.chenglong@gmail.com >
    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >
"""

import sys
import pickle
import pandas as pd
sys.path.append("../")
from param_config import config


def extract_aggr_feat(train, test):
    print("    Generating solo aggr...")
    # 单特征聚合信息, 按每个特征计算speed和duration的均值作为新的特征
    gby_cols = ['pickup_hour',
                'pickup_date',
                'pickup_dt_bin',
                'pickup_week_hour',
                'pickup_cluster',
                'dropoff_cluster']
    for gby_col in gby_cols:
        # gby即为按相应列的值分组后组内的均值
        gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']]
        gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
        # 按照相应列与数据集合并
        train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
        test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

    print("    Generating dual aggr...")
    # 多特征聚合信息, 按每个特征组合计算speed和count(id)的均值作为新的特征, 每组均值要考虑容量(>100)才认为具有代表性
    for gby_cols in [['center_lat_bin', 'center_long_bin'],
                     ['pickup_hour', 'center_lat_bin', 'center_long_bin'],
                     ['pickup_hour', 'pickup_cluster'],
                     ['pickup_hour', 'dropoff_cluster'],
                     ['pickup_cluster', 'dropoff_cluster']]:
        coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
        coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
        coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)

        coord_stats = coord_stats[coord_stats['id'] > 100]
        coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' % '_'.join(gby_cols)]

        train = pd.merge(train, coord_stats, how='left', on=gby_cols)
        test = pd.merge(test, coord_stats, how='left', on=gby_cols)

    print("    Generating count aggr...")
    # Count trips over 60min
    group_freq = '60min'
    train['pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
    test['pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

    df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
    df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
    df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
    train = train.merge(df_counts, on='id', how='left')
    test = test.merge(df_counts, on='id', how='left')

    print("    Generating dropoff aggr...")
    # Count how many trips are going to each cluster over time
    dropoff_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster']) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('dropoff_cluster').rolling('240min').mean() \
        .drop('dropoff_cluster', axis=1) \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

    train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)
    test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)

    print("    Generating pickup aggr...")
    # Count how many trips are going from each cluster over time
    df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
    pickup_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.TimeGrouper(group_freq), 'pickup_cluster']) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('pickup_cluster').rolling('240min').mean() \
        .drop('pickup_cluster', axis=1) \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_cluster_count'})

    train['pickup_cluster_count'] = train[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
    test['pickup_cluster_count'] = test[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
    return train, test

if __name__ == "__main__":
    # Load Data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)

    # Generate Features
    print("Generate aggregate features...")
    dfTrain, dfTest = extract_aggr_feat(dfTrain, dfTest)

    # Dump Data
    with open(config.processed_train_data_path, "wb") as f:
        pickle.dump(dfTrain, f, -1)
    with open(config.processed_test_data_path, "wb") as f:
        pickle.dump(dfTest, f, -1)
    print("Done.")
