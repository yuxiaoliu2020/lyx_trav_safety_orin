#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lyx
exec python $(rospack find trav_safety)/scripts/traversability_infer_node.py "$@"