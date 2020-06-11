#!/bin/bash
FILES=configs/neurips/ihdp1000/*
for f in $FILES
do
    screen -S "ihdp1000_eval" -d -m bash -c 'source activate_env.sh; CUDA_VISIBLE_DEVICES="" python evaluate.py '"$f"' 1; sleep 0.5s; '
    echo "************************************************"
done