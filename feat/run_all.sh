#/bin/bash
PYTHON=/home/poi/python27/bin/python

function log_print() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${1}"
}

log_print "preporcess"
$PYTHON ./0_preprocess.py
if [ $? -ne 0 ]; then
    log_print "preporcess fail"
    exit -1
fi

log_print "generate time based features"
$PYTHON ./1_gen_time_feat.py
if [ $? -ne 0 ]; then
    log_print "generate time based features fail"
    exit -1
fi

log_print "generate coordinate based features"
$PYTHON ./2_gen_coord_feat.py
if [ $? -ne 0 ]; then
    log_print "generate coordinate based features fail"
    exit -1
fi

log_print "generate distance based features"
$PYTHON ./3_gen_distance_feat.py
if [ $? -ne 0 ]; then
    log_print "generate distance based features fail"
    exit -1
fi

log_print "generate clustering based features"
$PYTHON ./4_gen_cluster_feat.py
if [ $? -ne 0 ]; then
    log_print "generate clustering based features fail"
    exit -1
fi

log_print "generate aggregating features"
$PYTHON ./5_gen_aggr_feat.py
if [ $? -ne 0 ]; then
    log_print "generate aggregating features fail"
    exit -1
fi

log_print "generate additional osrm features"
$PYTHON ./6_gen_osrm_feat.py
if [ $? -ne 0 ]; then
    log_print "generate additional osrm features fail"
    exit -1
fi

log_print "combine features"
$PYTHON ./feats_8_24.py
if [ $? -ne 0 ]; then
    log_print "combine features fail"
    exit -1
fi

