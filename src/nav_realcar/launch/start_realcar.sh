#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lyx
export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
roslaunch nav_realcar start_realcar.launch "$@"