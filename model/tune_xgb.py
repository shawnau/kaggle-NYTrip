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

# using xgb.cv
# dtrain = xgb.DMatrix(train.values,
#                      label=y,
#                      feature_names=train.columns.values)

# using xgb.train with sklearn
Xtr, Xv, ytr, yv = train_test_split(train.values, y, test_size=0.25, random_state=1992)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# grid search
xgb_pars = []
train_dict = {}
cv_log = []
for MCW in [5, 10]:
    for ETA in [0.01]:
        for CS in [0.6]: # 0.6 < 0.7 < 0.5
            for MD in [9, 10]: # 5 or 6 seems better to convergence
                for SS in [1.0]:
                    for LAMBDA in [2.0]:
                        xgb_pars.append({'min_child_weight': MCW,
                                         'eta': ETA,
                                         'colsample_bytree': CS,
                                         'max_depth': MD,
                                         'subsample': SS,
                                         'lambda': LAMBDA,
                                         'nthread': -1,
                                         'booster' : 'gbtree',
                                         'eval_metric': 'rmse',
                                         'silent': 1,
                                         'objective': 'reg:linear'})

# print('Start training...')
for xgb_par in xgb_pars:
    print(xgb_par)
    model = xgb.train(xgb_par,
                      dtrain,
                      50000,
                      evals=watchlist,  # for xgb.train
                      evals_result=train_dict, # for xgb.train
                      early_stopping_rounds=50,
                      maximize=False,
                      verbose_eval=500
                      # seed=1992  # for xgb.cv
                      )
    cv_log.append([xgb_par, copy.deepcopy(train_dict)])  # for xgb.cv
    s = "MCW-%.2f-MD-%d" % (xgb_par['min_child_weight'], xgb_par['max_depth'])
    print("Dump xgb model...", end="")
    with open("xgb_model_%s.pkl" % (s), "wb") as f:
        pickle.dump(model, f, -1)
    print("Done")


# dump log data
t = datetime.datetime.now()
s = "%s-%s-%s-%s-%s" % (t.year, t.month, t.day, t.hour, t.minute)
print("Dump tuning log...", end="")
with open("tune_log_%s.pkl" % (s), "wb") as f:
    pickle.dump(cv_log, f, -1)
print("Done")
