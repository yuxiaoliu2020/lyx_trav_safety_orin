#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lyx
export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
which python
roslaunch trav_safety trav_safety_realcar.launch "$@"