# coding: utf-8
# pylint: disable = invalid-name, C0111
import sys
import pickle
import json
import datetime
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
sys.path.append("../")
from param_config import config

# load or create your dataset
print('Load data...')
with open(config.final_train_data_path, "rb") as f:
    dfTrain = pickle.load(f)

train = dfTrain[[x for x in dfTrain.columns.values if x != 'log_trip_duration']]
y = dfTrain['log_trip_duration'].values

# split data into train and valid
Xtrain, Xvalid, ytrain, yvalid = train_test_split(train.values, y, test_size=0.2, random_state=1992)

# create dataset for lightgbm
lgb_train = lgb.Dataset(Xtrain, ytrain)
lgb_eval = lgb.Dataset(Xvalid, yvalid, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2_root'},
    'max_depth': -1,
    'num_leaves': 2048,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.9,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=15000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)

t = datetime.datetime.now()
s = "%s-%s-%s-%s-%s" % (t.year, t.month, t.day, t.hour, t.minute)

# dump model
print("Dump lgb model...", end="")
with open("lgb_model_%s.pkl" % (s), "wb") as f:
    pickle.dump(gbm, f, -1)
print("Done")
