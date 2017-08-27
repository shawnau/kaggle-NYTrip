import sys
import pickle
import pandas as pd
import xgboost as xgb
import copy
import datetime
from sklearn.model_selection import train_test_split
sys.path.append("../")
from param_config import config

print("Load data...")
with open(config.final_train_data_path, "rb") as f:
    dfTrain = pickle.load(f)

train = dfTrain[[x for x in dfTrain.columns.values if x != 'log_trip_duration']]
y = dfTrain['log_trip_duration'].values

# using xgb.train with sklearn
Xtr, Xv, ytr, yv = train_test_split(train.values, y, test_size=0.2, random_state=1992)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_par = {'min_child_weight': 20,
           'eta': 0.05,
           'colsample_bytree': 0.6,
           'max_depth': 5,
           'subsample': 1.0,
           'lambda': 2.0,
           'nthread': -1,
           'booster' : 'gbtree',
           'eval_metric': 'rmse',
           'silent': 1,
           'objective': 'reg:linear'}

# train model
train_dict = {}
model = xgb.train(xgb_par,
                  dtrain,
                  50000,
                  evals=watchlist,
                  evals_result=train_dict,
                  early_stopping_rounds=50,
                  maximize=False,
                  verbose_eval=1000
                  )

# dump log data
t = datetime.datetime.now()
s = "%s-%s-%s-%s-%s" % (t.year, t.month, t.day, t.hour, t.minute)

print("Dump tuning log...", end="")
with open("train_log_%s.pkl" % (s), "wb") as f:
    pickle.dump(train_dict, f, -1)
print("Done")

# dump model
print("Dump xgb model...", end="")
with open("xgb_model_%s.pkl" % (s), "wb") as f:
    pickle.dump(model, f, -1)
print("Done")