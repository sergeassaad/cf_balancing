#!/bin/bash
FILES=configs/neurips/ihdp100_trees/*
for f in $FILES
do
    screen -S "ihdp100" -d -m bash -c 'source activate_env.sh; CUDA_VISIBLE_DEVICES="" python cfr_param_search.py '"$f"' 1; sleep 0.5s; '
    echo "************************************************"
done