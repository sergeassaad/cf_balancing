#!/bin/bash

for p_idx in {1..50}
do
    screen -S "ihdp_$p_idx" -d -m bash -c 'source activate_env.sh; CUDA_VISIBLE_DEVICES="" python ablation.py configs/neurips/ihdp100.txt 2 "weight_scheme"; sleep 0.5s; '
    echo "************************************************"
done