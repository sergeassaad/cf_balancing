#!/bin/bash

for p_idx in {1..50}
do
    screen -S "toy$p_idx" -d -m bash -c 'source activate_env.sh; CUDA_VISIBLE_DEVICES="" python cfr_param_search.py configs/neurips/toy.txt 1000; sleep 0.5s; '
    echo "************************************************"
done
