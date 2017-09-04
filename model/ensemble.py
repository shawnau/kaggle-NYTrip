import numpy as np
import pandas as pd

print("Loading data...")
lgb_preds = pd.read_csv('lgb_model_2017-9-4-17-34.csv')
xgb_preds = pd.read_csv('xgb_model_MCW-5.00-MD-10.csv')

merged = pd.merge(xgb_preds, lgb_preds, on='id')
merged['trip_duration'] = (merged['trip_duration_x'] + merged['trip_duration_y']) / 2.0
merged = merged.drop(['trip_duration_x', 'trip_duration_y'], axis=1)

print("Dump merged data...")
merged[['id', 'trip_duration']].to_csv('merged.csv.gz', index=False, compression='gzip')
