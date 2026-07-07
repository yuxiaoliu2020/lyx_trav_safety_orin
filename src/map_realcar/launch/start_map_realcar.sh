#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lyx
export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
roslaunch map_realcar start_map_realcar.launch "$@"