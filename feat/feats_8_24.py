import sys
import pickle
import pandas as pd
sys.path.append("../")
from param_config import config


if __name__ == '__main__':
    feats = [
       # 'id',
       'vendor_id',
       # 'pickup_datetime',
       # 'dropoff_datetime',
       'passenger_count',
       'pickup_longitude',
       'pickup_latitude',
       'dropoff_longitude',
       'dropoff_latitude',
       'store_and_fwd_flag',
       # 'trip_duration',
       # 'log_trip_duration',
       # 'pickup_date',
       'pickup_weekday',
       'pickup_hour_weekofyear',
       'pickup_hour',
       'pickup_minute',
       'pickup_dt',
       'pickup_week_hour',
       'pickup_pca0',
       'pickup_pca1',
       'dropoff_pca0',
       'dropoff_pca1',
       'center_latitude',
       'center_longitude',
       # 'pickup_lat_bin',
       # 'pickup_long_bin',
       # 'center_lat_bin',
       # 'center_long_bin',
       # 'pickup_dt_bin',
       'distance_haversine',
       'distance_dummy_manhattan',
       'direction',
       'pca_manhattan',
       # 'avg_speed_h',
       # 'avg_speed_m',
       'pickup_cluster',
       'dropoff_cluster',
       'avg_speed_h_gby_pickup_hour',
       'avg_speed_m_gby_pickup_hour',
       'log_trip_duration_gby_pickup_hour',
       'avg_speed_h_gby_pickup_date',
       'avg_speed_m_gby_pickup_date',
       'log_trip_duration_gby_pickup_date',
       'avg_speed_h_gby_pickup_dt_bin',
       'avg_speed_m_gby_pickup_dt_bin',
       'log_trip_duration_gby_pickup_dt_bin',
       'avg_speed_h_gby_pickup_week_hour',
       'avg_speed_m_gby_pickup_week_hour',
       'log_trip_duration_gby_pickup_week_hour',
       'avg_speed_h_gby_pickup_cluster',
       'avg_speed_m_gby_pickup_cluster',
       'log_trip_duration_gby_pickup_cluster',
       'avg_speed_h_gby_dropoff_cluster',
       'avg_speed_m_gby_dropoff_cluster',
       'log_trip_duration_gby_dropoff_cluster',
       'avg_speed_h_center_lat_bin_center_long_bin',
       'cnt_center_lat_bin_center_long_bin',
       'avg_speed_h_pickup_hour_center_lat_bin_center_long_bin',
       'cnt_pickup_hour_center_lat_bin_center_long_bin',
       'avg_speed_h_pickup_hour_pickup_cluster',
       'cnt_pickup_hour_pickup_cluster',
       'avg_speed_h_pickup_hour_dropoff_cluster',
       'cnt_pickup_hour_dropoff_cluster',
       'avg_speed_h_pickup_cluster_dropoff_cluster',
       'cnt_pickup_cluster_dropoff_cluster',
       # 'pickup_datetime_group',
       'count_60min',
       'dropoff_cluster_count',
       'pickup_cluster_count',
       'total_distance',
       'total_travel_time',
       'number_of_steps'
       ]

    print "Load data..."
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = pickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = pickle.load(f)

    print "Dump final dataset...",
    with open(config.final_train_data_path, "wb") as f:
        pickle.dump(dfTrain[feats + ['log_trip_duration']], f, -1)
    with open(config.final_test_data_path, "wb") as f:
        pickle.dump(dfTest[feats], f, -1)
    print "Done."
