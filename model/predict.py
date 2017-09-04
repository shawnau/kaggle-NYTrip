# coding: utf-8
import sys
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import h2o
import lightgbm as lgb
sys.path.append("../")
from param_config import config
from utils import *

def dump_data(pandasDf, filename):
    print("Load raw data...")
    with open(config.processed_test_data_path, "rb") as f:
        dfTest_raw = pickle.load(f)

    print('Test shape OK.') if pandasDf.shape[0] == dfTest_raw.shape[0] else print('Oops')

    dfTest_raw['trip_duration'] = np.exp(pandasDf) - 1
    print('Dump data...')
    dfTest_raw[['id', 'trip_duration']].to_csv('%s.csv.gz'%filename, index=False, compression='gzip')
    print('Done')


print("Load data...")
with open(config.final_test_data_path, "rb") as f:
    dfTest = pickle.load(f)

# Train xgb model
print("Load xgb model...")
dtest = xgb.DMatrix(dfTest.values)
with open('xgb_model_MCW-5.00-MD-10.pkl', "rb") as f:
    model = pickle.load(f)
print("xgb predicting...")
xgb_predict = model.predict(dtest)

# Train h2o model
print("Load h2o model...")
h2o.init()
dtest = pandas_to_h2o(dfTest)
gbm = h2o.load_model('Grid_GBM_py_3_sid_99ed_model_python_1504245495640_1_model_0')
print("h2o predicting...")
h2o_predict = gbm.predict(dtest)
h2o_predict = h2o_to_pandas(h2o_predict)

# Train lightgbm model
print("Load lgb model...")
with open('lgb_model_2017-9-4-17-34.pkl', "rb") as f:
    lgb = pickle.load(f)
print("lgb predicting...")
lgb_predict = pd.DataFrame(lgb.predict(dfTest.values))

# dump data
dump_data(xgb_predict, 'xgb_model_MCW-5.00-MD-10.pkl')
dump_data(h2o_predict, 'Grid_GBM_py_3_sid_99ed_model_python_1504245495640_1_model_0')
dump_data(lgb_predict, 'lgb_model_2017-9-4-17-34')
