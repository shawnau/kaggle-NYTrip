# coding: utf-8
import sys
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
sys.path.append("../")
from param_config import config

print("Load data...")
with open(config.final_test_data_path, "rb") as f:
    dfTest = pickle.load(f)

dtest = xgb.DMatrix(dfTest.values)

print("Load model...")
with open('xgb_model_MCW-5.00-MD-10.pkl', "rb") as f:
    model = pickle.load(f)

print("Predicting...")
result = model.predict(dtest)

print("Load raw data...")
with open(config.processed_test_data_path, "rb") as f:
    dfTest_raw = pickle.load(f)

print('Test shape OK.') if result.shape[0] == dfTest_raw.shape[0] else print('Oops')

dfTest_raw['trip_duration'] = np.exp(result) - 1
print('Dump data...')
dfTest_raw[['id', 'trip_duration']].to_csv('xgb_model_MCW-5.00-MD-10.csv.gz', index=False, compression='gzip')
print('Done')
