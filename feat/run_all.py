
"""
__file__

    run_all.py

__description___
    
    This file generates all the features in one shot.

__author__

    xiaoxuan github.com/shawnau

"""

import os

#################
## Preprocesss ##
#################
#### preprocess data
cmd = "python ./0_preprocess.py"
os.system(cmd)

# #### generate kfold
# cmd = "python ./gen_kfold.py"
# os.system(cmd)

#######################
## Generate features ##
#######################
#### time based fratures
cmd = "python ./1_gen_time_feat.py"
os.system(cmd)

#### coordinate based features
cmd = "python ./2_gen_coord_feat.py"
os.system(cmd)

#### distance based features
cmd = "python ./3_gen_distance_feat.py"
os.system(cmd)

#### clustering based features
cmd = "python ./4_gen_cluster_feat.py"
os.system(cmd)

#### aggregating features
cmd = "python ./5_gen_aggr_feat.py"
os.system(cmd)

#### additional osrm features
cmd = "python ./6_gen_osrm_feat.py"
os.system(cmd)

#####################
## Combine Feature ##
#####################
#### combine features
cmd = "python ./feats_8_24.py"
os.system(cmd)