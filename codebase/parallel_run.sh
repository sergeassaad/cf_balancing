#!/bin/bash

for p_idx in {1..10}
do
    screen -S "par_causal_$p_idx" -d -m bash -c 'source activate_env.sh; python cfr_param_search.py configs/weighted_ihdp4.txt 100000; sleep 0.5s; '
    echo "************************************************"
done
