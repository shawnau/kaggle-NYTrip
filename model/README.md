# Tune single model (xgboost)
 - generate train data: run `../feat/run_all.py`
 - run `tune_xgb.py`
 - find best hyperparameters

# Train single model (xgboost)
 - generate train data(if not): run `../feat/run_all.py`
 - run `train_xgb.py`
 - model saved in `xgb_model_<DATETIME>.pkl`

# Stacking
 see `stack.py`, with test case

# To do list:
 - separate model parameter from training script
 - integrate with lightGBM or H20 model
 - hyperopt support
