
"""
__file__

    param_config.py

__description__

    This file provides global parameter configurations for the project.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

    Modified by xiaoxuan < xxuan@mail.ustc.edu.cn >

"""

import os
import numpy as np


############
## Config ##
############
class ParamConfig:
    def __init__(self):

#        # CV params
#        self.n_runs = 3
#        self.n_folds = 3
#        self.stratified_label = "query"

        # path
        self.data_folder = "../data"
        self.feat_folder = "../feat/pkl"
        self.original_train_data_path = "%s/train.csv" % self.data_folder
        self.original_test_data_path = "%s/test.csv" % self.data_folder
        self.processed_train_data_path = "%s/train.processed.csv.pkl" % self.feat_folder
        self.processed_test_data_path = "%s/test.processed.csv.pkl" % self.feat_folder
        self.final_train_data_path = "%s/train.final.csv.pkl" % self.feat_folder
        self.final_test_data_path = "%s/test.final.csv.pkl" % self.feat_folder

        # create feat folder
        if not os.path.exists(self.feat_folder):
            os.makedirs(self.feat_folder)

#        # creat folder for the training and testing feat
#        if not os.path.exists("%s/All" % self.feat_folder):
#            os.makedirs("%s/All" % self.feat_folder)
#
#        # creat folder for each run and fold
#        for run in range(1, self.n_runs+1):
#            for fold in range(1, self.n_folds+1):
#                path = "%s/Run%d/Fold%d" % (self.feat_folder, run, fold)
#                if not os.path.exists(path):
#                    os.makedirs(path)


# initialize a param config
config = ParamConfig()
