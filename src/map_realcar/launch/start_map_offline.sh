#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wayfaster
export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
roslaunch map_realcar start_map_offline.launch "$@"