# coding: utf-8

import sys
import pickle
import pandas as pd
import copy
import datetime
sys.path.append("../")
from param_config import config
from utils import *

import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

h2o.init(max_mem_size='20g')
print("Load data...")
with open(config.final_train_data_path, "rb") as f:
    dfTrain = pickle.load(f)

trainDf = pandas_to_h2o(dfTrain)

predict = [x for x in trainDf.columns if x != 'log_trip_duration']
response = 'log_trip_duration'

train, valid = trainDf.split_frame(ratios=[0.8], seed=1992)

gbm_param = {
    'stopping_metric': 'rmse',
    'stopping_rounds': 2
}

gd_params = {
    'ntrees': [20000],
    'sample_rate': [0.8, 0.9, 1.0],
    'col_sample_rate': [0.6, 0.8, 1.0],
    'learn_rate': [0.01],
    'max_depth': [10]
}

grid_search = H2OGridSearch(H2OGradientBoostingEstimator(**gbm_param), hyper_params=gd_params)
grid_search.train(x=predict, y=response, training_frame=train, validation_frame=valid)
# gbm.train(x=predict, y=response, training_frame=train, validation_frame=valid)

for model in grid_search:
    h2o.save_model(model=model, path="model/", force=True)

print(grid_search.get_grid(sort_by='rmse', decreasing=True))
